# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from plotly.offline import init_notebook_mode, iplot, plot

import plotly.express as px

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt



import warnings            

warnings.filterwarnings("ignore") 
covid_data = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')

covid_data.head()
print('Total tweets in this data: {}'.format(covid_data.shape[0]))

print('Total Unique Users in this data: {}'.format(covid_data['user_name'].nunique()))
# info of the data



covid_data.info()
covid_data['country_name'] = covid_data['user_location'].str.split(',').str[-1]

covid_data['only_date'] = pd.to_datetime(covid_data['date']).dt.date
# let's see top 15 users by no. of tweets



user_analysis = pd.DataFrame(covid_data['user_name'].value_counts().sort_values(ascending=False))

user_analysis = user_analysis.rename(columns={'user_name':'count'})



trace = go.Bar(x = user_analysis.index[:15],

              y = user_analysis['count'][:15],

              marker = dict(color='rgba(255,155,128,0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top 15 user by no. of tweets",

                  xaxis=dict(title='User Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency of tweets',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# let's see top 15 users by no. of tweets



location_analysis = pd.DataFrame(covid_data['user_location'].value_counts().sort_values(ascending=False))

location_analysis = location_analysis.rename(columns={'user_location':'count'})



trace = go.Bar(x = location_analysis.index[:15],

              y = location_analysis['count'][:15],

              marker = dict(color='rgba(125, 215, 180, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top 15 Location by no. of tweets",

                  xaxis=dict(title='Location Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency of tweets',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
data = {

   "values": location_analysis['count'][:15],

   "labels": location_analysis.index[:15],

   "domain": {"column": 0},

   "name": "Location Name",

   "hoverinfo":"label+percent+name",

   "hole": .4,

   "type": "pie"

}

layout = go.Layout(

   {

      "title":"Location Ratio",

}

)



data = [data]

fig = go.Figure(data = data, layout = layout)

iplot(fig)

tweet_analysis = pd.DataFrame(covid_data['only_date'].value_counts())

tweet_analysis = tweet_analysis.rename(columns={'only_date':'count'})



trace = go.Bar(x = tweet_analysis.index,

              y = tweet_analysis['count'],

              marker = dict(color='rgba(150, 200, 100, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(barmode='group',

                  title="Date wise no. of tweets",

                  xaxis=dict(title='Date',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency of tweets',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# top source 

source_analysis = pd.DataFrame(covid_data['source'].value_counts().sort_values(ascending=False))

source_analysis = source_analysis.rename(columns={'source':'count'})



trace = go.Bar(x = source_analysis.index[:10],

              y = source_analysis['count'][:10],

              marker = dict(color='rgba(150, 125, 180, 0.5)',

              line = dict(color='rgb(0,0,0)', width=1.5)))



layout = go.Layout(title="Top 10 Sources by no. of tweets",

                  xaxis=dict(title='Source Name',zeroline= False,

                         gridcolor='rgb(183,183,183)',showline=True),

                  yaxis=dict(title='Frequency of tweets',zeroline= False,

                            gridcolor='rgb(183,183,183)',showline=True),

                  font=dict(family='Courier New, monospace', size=12, color='rgb(0,0,0)')

)

data = [trace]

fig = go.Figure(data = data, layout = layout)

iplot(fig)
data = {

   "values": source_analysis['count'][:15],

   "labels": source_analysis.index[:15],

   "domain": {"column": 0},

   "name": "Source Name",

   "hoverinfo":"label+percent+name",

   "hole": .4,

   "type": "pie"

}

layout = go.Layout(

   {

      "title":"Source Ratio of Top 15 sources",

}

)

data = [data]

fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    autosize=False,

    width=1200,

    height=700,)

iplot(fig)
def wordcloud(string):

    wc = WordCloud(width=800,height=500,mask=None,random_state=21, max_font_size=110,stopwords=stop_words).generate(string)

    fig=plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(wc)
stop_words=set(STOPWORDS)

country_string = " ".join(covid_data['country_name'].astype('str'))

source_string = " ".join(covid_data['source'].astype('str'))

text_string = " ".join(covid_data['text'])

description_string = " ".join(covid_data['user_description'].astype('str'))

hastage_string = " ".join(covid_data['hashtags'].astype('str'))

location_string = " ".join(covid_data['user_location'].astype('str'))
wordcloud(country_string)
wordcloud(source_string)
wordcloud(text_string)
wordcloud(description_string)
wordcloud(hastage_string)
wordcloud(location_string)
sentiment_data = pd.read_csv('/kaggle/input/twitterdata/finalSentimentdata2.csv')
sentiment_data.head()
sentiment_data = sentiment_data.drop(columns=['Unnamed: 0'])
sentiment_data['sentiment'].unique()
import re

import string

def remove_punc(text):

    # Dealing with Punctuation

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



sentiment_data['text'] = sentiment_data['text'].apply(remove_punc)
from nltk import stem

from nltk.corpus import stopwords

stemmer = stem.SnowballStemmer('english')

stopwords = set(stopwords.words('english'))



def alternative_review_messages(msg):

    # converting messages to lowercase

    msg = msg.lower()

    # removing stopwords

    msg = [word for word in msg.split() if word not in stopwords]

    # using a stemmer

    msg = " ".join([stemmer.stem(word) for word in msg])

    return msg



sentiment_data['text'] = sentiment_data['text'].apply(alternative_review_messages)
SEED = 2000

x_train, x_validation, y_train, y_validation = train_test_split(sentiment_data['text'], sentiment_data['sentiment'], 

                                                                test_size=.2, random_state=SEED)
from time import time

def prediction(pipeline, x_train, y_train,testtext):

    t0 = time()

    sentiment_fit = pipeline.fit(x_train, y_train)

    y_pred = sentiment_fit.predict(testtext)

    return y_pred
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import RidgeClassifier



vectorizer=TfidfVectorizer()

checker_pipeline = Pipeline([

            ('vectorizer', vectorizer),

            ('classifier', RidgeClassifier())

        ])

vectorizer.set_params(stop_words=None, max_features=10000, ngram_range=(1,4))

prediction=prediction(checker_pipeline,x_train, y_train,x_validation)
from sklearn.metrics import accuracy_score

def acc_summary(pipeline, x_train, y_train, x_test, y_test):

    t0 = time()

    sentiment_fit = pipeline.fit(x_train, y_train)

    y_pred = sentiment_fit.predict(x_test)

    train_test_time = time() - t0

    accuracy = accuracy_score(y_test, y_pred)

    print("accuracy score: {0:.2f}%".format(accuracy*100))

    print("train and test time: {0:.2f}s".format(train_test_time))

    print("-"*80)

    return accuracy, train_test_time

clf_acc = acc_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
from sklearn.svm import SVC

def prediction2(pipeline, x_train, y_train,testtext):

    t0 = time()

    sentiment_fit = pipeline.fit(x_train, y_train)

    y_pred = sentiment_fit.predict(testtext)

    return y_pred

checker_pipeline2 = Pipeline([

            ('vectorizer', vectorizer),

            ('classifier', SVC(C=1000))

        ])

vectorizer.set_params(stop_words=None, max_features=10000, ngram_range=(1,4))

prediction=prediction2(checker_pipeline2,x_train, y_train,x_validation)
clf_acc = acc_summary(checker_pipeline2, x_train, y_train, x_validation, y_validation)