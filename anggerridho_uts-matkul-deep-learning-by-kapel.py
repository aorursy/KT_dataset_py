!pip install tweepy
import numpy as np

from numpy.random import randint

import pandas as pd

import re

import tweepy

from tweepy import OAuthHandler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cluster import KMeans

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg'
consumer_key = '6XoXBKpBmROf4nx4CQvWbR21i'

consumer_secret = 'aeQTqli8Ee8VLljGgmMCzShpqwpYqGGwGWv0lnNdFgSLZaLCQp'

access_token = '809736756031463426-cqhABrYcrflkdAhVs4ee93KFFcLPRcI'

access_token_secret = 's6xOJzWvQyIOHWQZkk7HxK9o1tzaPMhuzlyYN5ZWcHzxj'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)



api = tweepy.API(auth)
# The search term you want to find

query = "Pandji Pragiwaksono"

# Language code (follows ISO 639-1 standards)

language = "id"



# Calling the user_timeline function with our parameters

results = api.search(q=query, lang=language, count=5)



# foreach through all tweets pulled

message = []

for tweet in results:

    message.append(tweet.text)



data = pd.DataFrame({'Tweet': message})

print(data.shape)

data.head()
def clean_text(text):

    text = re.sub(r'[^a-z]', ' ', text.lower())

    text = re.sub(r'\s+', ' ', text)

    return text



data.Tweet = [clean_text(i) for i in data.Tweet]

data.head()
vectorizer = CountVectorizer(min_df=2, binary=True)

vectorizer.fit(data.Tweet)
data_vect = pd.DataFrame(vectorizer.transform(data.Tweet).toarray(), columns=vectorizer.get_feature_names())

print(data_vect.shape)

data_vect.head()
all_text = [i for i in data.Tweet]

all_text = ' '.join(all_text)
wordcloud = WordCloud().generate(all_text)

# Display the generated image:

plt.figure(figsize=(10, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
kmeans = KMeans(n_clusters=2)

kmeans.fit(data_vect)
data['sentiment'] = kmeans.predict(data_vect)

data.head()