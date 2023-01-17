# The basic imports
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
from tqdm import tqdm
from textblob import TextBlob
example = "This is a wonderful product. I got amazing results by using it"
blob = TextBlob(example) # create the TextBlob object for this example
blob.sentiment
example = "This is a terrible product. I would not recommend using it"
blob = TextBlob(example)
blob.sentiment
# Polarity
blob.sentiment.polarity
# Subjectivity
blob.sentiment.subjectivity
example = "Jupiter is the largest planet in the Solar System"
blob = TextBlob(example)
blob.sentiment
train = pd.read_csv('../input/drugsComTrain_raw.csv')

# Inspect first 5 rows
train.head(5)
train.shape
reviews = train["review"]
print(reviews[:10]) # First 10
sentiments = []
for review in tqdm(reviews):
    blob = TextBlob(review)
    sentiments += [blob.sentiment.polarity]
train["sentiment"] = sentiments
train.sample(10)
np.corrcoef(train["rating"], train["sentiment"])
import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x=np.array(train["rating"]),y=np.array(train["sentiment"]))
plt.xlabel("Rating")
plt.ylabel("Sentiment")
plt.title("Sentiment vs Ratings")
plt.show()