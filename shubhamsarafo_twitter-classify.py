# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
all_tweets = pd.read_json('/kaggle/input/twitter-classification/random_tweets.json',lines =True)
#total number of tweets in the dataset

print(len(all_tweets))

print('\n')

#columns and features

print(all_tweets.columns)

print('\n')

#text of the first tweet in the dataset

print(all_tweets.loc[0]['text'])
#feature user is a dictionary

print(all_tweets.loc[0]['user'])
#just the location  of user

print(all_tweets.loc[0]['user']['location'])
#checking re-tweet count to determine viral_tweets

print(all_tweets['retweet_count'])
#we will use the median number forretweet count



median_retweets = all_tweets.retweet_count.median()



#if retweet count is more than the median count then we will classify them as 1 otherwise 0



all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > median_retweets , 1 , 0)

print(all_tweets['is_viral'])
### making FEATURES

# we want to know what makes the tweet viral; it can be the length of the tweet,can be how many

# hashtags the tweet has



all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']),axis=1)

print(all_tweets['tweet_length'])
#follower_count feature

all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'],axis=1)

#and same for firends_count

all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'],axis=1)
# can use number of hashtags in a tweet.  (using '#' count)

# the number of links in a tweet .  (using 'http' count)

# the number of words in a tweet . (using split)

# the avrg number of words in the tweet.
all_tweets['hashtag_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'),axis=1)

all_tweets['http_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('http'),axis=1)
#get rid of the data which is not relevant

labels = all_tweets['is_viral']

data = all_tweets[['tweet_length','followers_count','friends_count','hashtag_count','http_count']]
from sklearn.preprocessing import scale



scaled_data = scale(data , axis=0)
print(scaled_data)
from sklearn.model_selection import train_test_split
train_data , test_data ,train_labels , test_labels = train_test_split(data , labels , test_size = 0.2 , random_state = 101)
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(train_data,train_labels)
Knn.score(test_data,test_labels)
#we gonna chose a valid K which will increase our score
scores = []

for k in range(1,200):

    Knn = KNeighborsClassifier(n_neighbors=k)

    Knn.fit(train_data,train_labels)

    scores.append(Knn.score(test_data,test_labels))

plt.plot(scores)

plt.show()  
# we will chose our k somewhere between 30 - 38 to achieve 60% 

# we need to work more on the features to achive more accuracy but the machine is still  better than 50 - 50
# we are going to classify tweets or any sentence wether it came from new_york , london , paris
new_york_tweets = pd.read_json('/kaggle/input/twitter-classification/new_york.json',lines=True)
#total number of tweets in the dataset

print(len(new_york_tweets))

print('\n')

#columns and features

print(new_york_tweets.columns)

print('\n')

#text of the 13th tweet in the dataset

print(new_york_tweets.loc[12]['text'])
london_tweets = pd.read_json('/kaggle/input/twitter-classification/london.json',lines=True) 

paris_tweets = pd.read_json('/kaggle/input/twitter-classification/paris.json',lines=True)
print('New York')

print(len(new_york_tweets))

print('London')

print(len(london_tweets))

print('Paris')

print(len(paris_tweets))
new_york_text = new_york_tweets['text'].tolist()

london_text = london_tweets['text'].tolist()

paris_text = paris_tweets['text'].tolist()



df_tweet = new_york_text + london_text + paris_text

y = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)
X_train ,X_test , y_train , y_test =  train_test_split(df_tweet , y , test_size=0.2, random_state=1)
print(len(X_train))

print(len(X_test))
# we need CountVector

from sklearn.feature_extraction.text import CountVectorizer

counter = CountVectorizer()
counter.fit(X_train)

train_count = counter.transform(X_train)

test_count = counter.transform(X_test)
print(X_train[3])

print(train_count[3])
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(train_count,y_train)
predictions = mnb.predict(test_count)
print(predictions)
from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))
tweet = 'Hello ! My name is DjSarafO'

N_tweet = 'I live in Manhattan'
z = counter.transform([tweet])

p = counter.transform([N_tweet])

print(mnb.predict(z))

print(mnb.predict(p))