import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import random
import numpy as np

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Read the dataset into a pandas DataFrame
# Replace 'your_dataset.csv' with your actual file
df = pd.read_csv("Raw.csv", encoding='latin1')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# Assuming 'description' is the column with car descriptions
X_tfidf = tfidf_vectorizer.fit_transform(df['Description'])

# Analyze sentiments and apply a threshold
threshold = 0.8  # Adjust this threshold based on your preference

algorithm_priority_scores = {}

for car_name, description in zip(df["CAR_Name"], df["Description"]):
    # Ensure description is a string
    description = str(description)

    # Tokenize the description
    tokens = tokenizer(description, return_tensors="pt",
                       padding=True, max_length=512, truncation=True)

    # Make a forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits

    # Normalize the logits between 0 and 1 using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

    # Calculate an overall sentiment score
    # Assuming the model output for positive sentiment is at index 1
    overall_sentiment = probabilities[1].item()

    # Classify sentiment based on the threshold
    sentiment = 'positive' if overall_sentiment >= threshold else 'negative'

    algorithm_priority_scores[car_name] = overall_sentiment

# Identifying the car with the highest sentiment score
best_car = max(algorithm_priority_scores, key=algorithm_priority_scores.get)

# Printing the car with the best sentiment
print("Car with the best sentiment:")
print(best_car)

# Preparing the data for visualization
cars = list(algorithm_priority_scores.keys())
sentiments = list(algorithm_priority_scores.values())

# Convert cars to strings
cars = [str(car) for car in cars]

# Creating a bar plot to visualize the sentiments of cars
plt.figure(figsize=(10, 5))
plt.bar(cars, sentiments, label='Sentiments')
plt.xlabel("Car")
plt.ylabel("Positive Sentiment Probability")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Creating a line plot to visualize the sentiments of cars
plt.figure(figsize=(10, 5))
plt.plot(cars, sentiments, marker='o', linestyle='-',
         color='b', label='Sentiments')
plt.xlabel("Car")
plt.ylabel("Positive Sentiment Probability")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print sentiment score for each car
for car_name, description in zip(df["CAR_Name"], df["Description"]):
    if car_name == 'nan':
        continue
    # Ensure description is a string
    description = str(description)

    # Tokenize the description
    tokens = tokenizer([description], return_tensors="pt",
                       max_length=512, truncation=True)

    # Make a forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits

    # Normalize the logits between 0 and 1 using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

    # Calculate an overall sentiment score
    overall_sentiment = probabilities[1].item()

    # Classify sentiment based on the adjusted threshold
    sentiment = 'positive' if overall_sentiment >= threshold else 'negative'

    # Print sentiment score for each car
    print(f"{car_name}: {overall_sentiment} ({sentiment} sentiment)")

# Mapping sentiment scores to emojis


def map_to_emoji(score):
    if score >= 0.6:
        return "😄"  # High positive
    elif 0.4 <= score < 0.6:
        return "😐"  # Neutral
    else:
        return "😞"  # Negativ

# Print sentiment score with emojis for each car
for car_name, score in algorithm_priority_scores.items():
    sentiment = 'positive' if score >= threshold else 'negative'
    emoji = map_to_emoji(score)
    print(f"{car_name}: {score} ({sentiment} sentiment) {emoji}")

# Creating a scatter plot to visualize the sentiments of cars
plt.figure(figsize=(10, 5))
plt.scatter(cars, sentiments, marker='o', color='b', label='Sentiments')
plt.xlabel("Car")
plt.ylabel("Positive Sentiment Probability")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Creating a pie chart to visualize the sentiments of cars
plt.figure(figsize=(10, 5))
plt.pie(sentiments, labels=cars, autopct='%1.1f%%')
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.show()

# Creating a box plot to visualize the sentiments of cars
plt.figure(figsize=(10, 5))
plt.boxplot(sentiments)
plt.xlabel("Car")
plt.ylabel("Positive Sentiment Probability")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()