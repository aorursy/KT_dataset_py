import pandas as pd



all_tweets = pd.read_json("../input/random_tweets.json", lines=True)



print(len(all_tweets))

print(all_tweets.columns)

print(all_tweets.loc[0]['text'])



#Print the user here and the user's location here.
print(all_tweets.loc[0]['user'])

print(all_tweets.loc[0]['user']['location'])
import numpy as np

retweet_count_median = all_tweets["retweet_count"].median()

print(retweet_count_median)

all_tweets['is_viral'] = all_tweets['retweet_count'].apply(lambda x: 1 if x>=retweet_count_median else 0)

all_tweets['is_viral'].value_counts()
all_tweets.head(10)
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)

all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)

all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
from sklearn.preprocessing import scale



labels = all_tweets['is_viral']

data = all_tweets[['tweet_length', 'followers_count', 'friends_count']]

scaled_data = scale(data, axis=0)

print(scaled_data[0])
from sklearn.model_selection import train_test_split



train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size=0.2, random_state=1)
from sklearn.neighbors import KNeighborsClassifier



classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(train_data, train_labels)

print(classifier.score(test_data, test_labels))

import matplotlib.pyplot as plt



scores = []

ks = range(1,201)

for k in ks:

    classifier = KNeighborsClassifier(n_neighbors=k)

    classifier.fit(train_data, train_labels)

    scores.append(classifier.score(test_data, test_labels))

plt.plot(ks, scores)

plt.show()