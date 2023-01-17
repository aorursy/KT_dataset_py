import pandas as pd

import numpy as np

from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer
ratings = pd.read_excel("/kaggle/input/parisdata/rewiews808K.xlsx")
ratings.head()
blob = TextBlob(ratings['text'].iloc[0])

blob.tags
testimonial = TextBlob("terrible to use. What stupid!")
for sentence in blob.sentences:

    print(sentence, sentence.sentiment.polarity)
blob.sentiment
def get_sentiment(x):

    blob = TextBlob(x)

    return blob.sentiment.polarity
ratings['text'] = ratings['text'].apply(lambda x: str(x))
wdummy = ratings['text'].apply(lambda x: get_sentiment(x))
wdummy.shape
ratings['sentiment'] = wdummy
ratings.head()
ratings_copy = ratings
ratings_copy['sentiment'] = ratings_copy['sentiment'] + 1

ratings_copy['sentiment'] = ratings_copy['sentiment'] / 2
ratings_copy.tail()
ratings.to_csv("ratings.csv", index=None)

ratings_copy.to_csv("ratings_copy.csv", index=None)