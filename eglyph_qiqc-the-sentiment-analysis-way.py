# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk #for NLP processing and sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer #for calculating sentiment scores
from textblob import TextBlob #another approach of sentiment analysis
import plotly.plotly as py #For interactive Data Visualization
import plotly.graph_objs as go #For interactive Data Visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")



train_df.head()

test_df.head()

train1_df = train_df[train_df["target"]==1]
train0_df = train_df[train_df["target"]==0]
train1_df.head()
print("Insincere Group shape : ", train1_df.shape)
print("Sincere Group shape : ", train0_df.shape)

train0_df.head()
train1_df.head()
train1_df[['polarity','subjectivity']] = train1_df['question_text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
print(train1_df['polarity'].mean())
print(train1_df['subjectivity'].mean())
hist = train1_df['polarity'].hist()
hist = train1_df['subjectivity'].hist()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
train1_df['sentiment_scores'] = train1_df['question_text'].apply(sid.polarity_scores)
train1_df['sentiment'] = train1_df['sentiment_scores'].apply(lambda x: x['compound'])
hist = train1_df['sentiment'].hist()
print(train1_df['sentiment'].mean())