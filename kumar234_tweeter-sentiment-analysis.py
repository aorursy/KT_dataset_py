# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt
import missingno as ms
% matplotlib inline

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/twitter-hate-speech/train_E6oV3lV.csv')
test_data = pd.read_csv('../input/twitter-hate-speech/test_tweets_anuFYb8.csv')
train_data.shape, test_data.shape
train_data.head()
train_data.info()
train_data['label'].value_counts()

test_data.head()
#cleaning the data

def drop_features(features,data):
    data.drop(features,inplace=True,axis=1)
import re
## example ## 
re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ","ouch...junior is angryð#got7 #junior #yugyo..., @user")
def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())
train_data['processed_tweets'] = train_data['tweet'].apply(process_tweet)
train_data.head(10)
drop_features(['id','tweet'],train_data)
train_data.info()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data["processed_tweets"], train_data["label"], test_size = 0.2, random_state = 42)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)
print(x_train_counts.shape)
print(x_train_tfidf.shape)
x_train_counts
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)
print(x_test_counts.shape)
print(x_test_tfidf.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500)
model.fit(x_train_tfidf,y_train)
predictions = model.predict(x_test_tfidf)
from sklearn.metrics import confusion_matrix,f1_score
confusion_matrix(y_test,predictions)
#tp=5904
#tn = 237
#fn = 33
#fp = 219

#precision = tp/(tp+fp)
#recall = tp/(tp+fn)
#f1score = 2 * (recall * precision) / (recall + precision)
#f1score
f1_score(y_test,predictions)
predictions
test_data.info()
test_data['processed_tweet'] = test_data['tweet'].apply(process_tweet)
test_data.head()

drop_features(['tweet'],test_data)
train_counts = count_vect.fit_transform(train_data['processed_tweets'])
test_counts = count_vect.transform(test_data['processed_tweet'])
print(train_counts.shape)
print(test_counts.shape)
train_tfidf = transformer.fit_transform(train_counts)
test_tfidf = transformer.transform(test_counts)

print(train_tfidf.shape)
print(test_tfidf.shape)

model.fit(train_tfidf,train_data['label'])
predictions = model.predict(test_tfidf)
final_result = pd.DataFrame({'id':test_data['id'],'label':predictions})
final_result.to_csv('Output.csv',index=False)
final_result.head()