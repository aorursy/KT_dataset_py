import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.metrics import classification_report

print(os.listdir("../input"))

tweets = pd.read_csv("../input/Tweets.csv")
tweets = tweets.reindex(np.random.permutation(tweets.index))
tweets.head()
tweets.count()
del tweets['airline_sentiment_confidence']
del tweets['negativereason_confidence']
del tweets['airline_sentiment_gold']
del tweets['name']
del tweets['negativereason']
del tweets['negativereason_gold']
del tweets['retweet_count']
del tweets['tweet_coord']
del tweets['tweet_created']
del tweets['tweet_location']
del tweets['user_timezone']
tweets.head()
pd.value_counts(tweets['airline_sentiment'].values, sort = False)
pd.value_counts(tweets['airline'].values, sort = False)
from sklearn.feature_extraction.text import CountVectorizer
# create the transform
cv = CountVectorizer()
# tokenize and build vocab
cv.fit(tweets.text)

len(cv.vocabulary_)

docTerms = cv.fit_transform(tweets.text)
print(type(docTerms))
print(docTerms.shape)
%matplotlib inline
import matplotlib.pyplot as plt
plt.spy(docTerms,markersize=.25)

X_train, X_test, y_train, y_test = train_test_split(docTerms, tweets['airline_sentiment'].values, test_size = .7, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix
print(classification_report(y_test, y_pred))
tweet = ["late flight rude dirty refund wait"]
t = cv.transform(tweet)
print(tweet, " is ", LogReg.predict(t))
tweet = ["great fun nice good"]
t = cv.transform(tweet)
print(tweet, " is ", LogReg.predict(t))
tweet = ["bad"]
t = cv.transform(tweet)
print(tweet, " is ", LogReg.predict(t))
tweet = ["bad bad bad bad bad "]
t = cv.transform(tweet)
print(tweet, " is ", LogReg.predict(t))
tweet = ["should have taken the train"]
t = cv.transform(tweet)
print(tweet, " is ", LogReg.predict(t))