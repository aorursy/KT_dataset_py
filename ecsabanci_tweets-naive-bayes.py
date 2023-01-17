import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
new_york_tweets = pd.read_json("/kaggle/input/twitter-classification/new_york.json",lines=True)
print(len(new_york_tweets))
print(new_york_tweets.columns)
print(new_york_tweets.loc[12]["text"])
london_tweets = pd.read_json("/kaggle/input/twitter-classification/london.json",lines=True)
print(len(london_tweets))
paris_tweets = pd.read_json("/kaggle/input/twitter-classification/paris.json",lines=True)
print(len(paris_tweets))
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

all_tweets = new_york_text + london_text + paris_text
labels = [0]*len(new_york_text) + [1]*len(london_text) + [2]*len(paris_text)
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels  =train_test_split(all_tweets,labels, 
                                                                    test_size = 0.2, 
                                                                    random_state = 1)

print(len(train_data))
print(len(test_data))
from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

print(train_data[3])
print(train_counts[3])

# there is two "bye" and two "saying"...
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)
print(predictions)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(test_labels,predictions)
confusion_matrix(test_labels,predictions)
tweet = "je taime"
tweet_counts = counter.transform([tweet])
classifier.predict(tweet_counts)
tweets = pd.read_json("/kaggle/input/twitter-classification/random_tweets.json",lines=True)

print(len(tweets))
print(tweets.columns)
print(tweets.loc[0]["text"])

# each tweets are stores these columns....
tweets.loc[0]["user"]["location"]
median_retweets = tweets["retweet_count"].median()
tweets["is_viral"] = np.where(tweets["retweet_count"] > median_retweets, 1, 0 )
tweets["is_viral"] # if its 1 that means viral if not that means not viral....
tweets["tweet_len"] = tweets.apply(lambda tweet: len(tweet["text"]), axis = 1)
tweets["tweet_len"]
tweets["follower_count"] = tweets.apply(lambda tweet: tweet["user"]["followers_count"], axis = 1)
tweets["friends_count"] = tweets.apply(lambda tweet: tweet["user"]["friends_count"], axis = 1)
tweets["follower_count"]
tweets["hashtag_count"] = tweets.apply(lambda tweet: tweet["text"].count("#"), axis = 1)
tweets["hashtag_count"][0:15]
labels = tweets["is_viral"]

data = tweets[["tweet_len","follower_count","friends_count","hashtag_count"]]

from sklearn.preprocessing import scale

# we have to scale our data because if we dont number of 3 followers_count has a same impact on the prediction
# with number of 3 hashtag_count. we should avoid from this conflict.....

scaled_data = scale(data,axis=0)
scaled_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# lets take a look our score with scaled data......

train_data, test_data, train_labels, test_labels = train_test_split(scaled_data,labels,
                                                                   test_size=0.2,
                                                                   random_state=1)
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_data,train_labels)
classifier.score(test_data,test_labels)
# lets use our data(not scaled).....

train_data, test_data, train_labels, test_labels = train_test_split(data,labels,
                                                                   test_size=0.2,
                                                                   random_state=1)
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_data,train_labels)
classifier.score(test_data,test_labels)
import matplotlib.pyplot as plt
scores = []

for k in range(1,200):
    
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data,train_labels)
    scores.append(classifier.score(test_data,test_labels))
    
plt.plot(scores)
plt.show()
# when k ups too much underfitting starts.....