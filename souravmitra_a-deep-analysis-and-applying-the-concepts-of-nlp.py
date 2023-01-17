import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import plotly as py

from plotly.graph_objs import graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot

import cufflinks as cf
init_notebook_mode(connected = True)

cf.go_offline(connected = True)
tweet_data = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")
tweet_data.info()
tweet_data.drop(['airline_sentiment_gold', 'negativereason_gold', 'tweet_coord'], axis = 1, inplace = True)
tweet_data.info()
data = go.Pie(values = tweet_data['airline'].value_counts().values, 

              labels = tweet_data['airline'].value_counts().index.values, hole = 0.3)



map_airlinecount = go.Figure(data = data)
map_airlinecount
tweet_data['airline_sentiment'].iplot(kind = 'histogram')
data1 = go.Bar(x = tweet_data['negativereason'].value_counts().index.values, 

               y = tweet_data['negativereason'].value_counts().values,)



map_negativereason1 = go.Figure(data = data1)
map_negativereason1
data2 = go.Pie(values = tweet_data['negativereason'].value_counts().values,

               labels = tweet_data['negativereason'].value_counts().index.values,

               hole = 0.3)



map_negativereason2 = go.Figure(data = data2)
map_negativereason2
def negative_sentiment_plot(airline) :

    df = tweet_data[tweet_data['airline'] == airline]

    count = dict(df['negativereason'].value_counts())

    count1 = list(df['negativereason'].value_counts())

    reasons = list(df['negativereason'].unique())

    reasons = [x for x in reasons if str(x) != 'nan']

    df_reason = pd.DataFrame({'Reasons' : reasons})

    df_reason['count']=df_reason['Reasons'].apply(lambda x: count[x])

    return df_reason

    #plt.figure(figsize = (30,12))

    #plt.bar(reasons, count1)
negative_sentiment_plot('United')
import nltk

import re

from nltk.corpus import stopwords

import string
def clean_tweet(raw_tweet) :

    clean_tweet = re.sub('[^a-zA-Z]', ' ', raw_tweet)

    clean_tweet = clean_tweet.lower().split()

    clean_tweet = [x for x in clean_tweet if x not in stopwords.words('english')]

    clean_tweet = ' '.join(clean_tweet)

    return clean_tweet
tweet_data['clean_tweet'] = tweet_data['text'].apply(lambda x : clean_tweet(x))
tweet_data['rating'] = tweet_data['airline_sentiment'].apply(lambda x : 0 if x == 'negative' else 1)
tweet_ML_data = tweet_data[['clean_tweet', 'rating']]
tweet_ML_data.info()
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
x = tweet_ML_data['clean_tweet']

y = tweet_ML_data['rating']
countvectorizer = CountVectorizer()
countvectorizermatrix = countvectorizer.fit_transform(x)
tfidf = TfidfTransformer()
tfidfmatrix = tfidf.fit_transform(countvectorizermatrix)
x_train, x_test, y_train, y_test = train_test_split(tfidfmatrix, y, test_size = 0.3, random_state = 101)
model = MultinomialNB()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators = 200)
model1.fit(x_train, y_train)
predictions1 = model1.predict(x_test)
print(classification_report(y_test, predictions1))

print(confusion_matrix(y_test, predictions1))