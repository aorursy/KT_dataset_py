import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns

import pandas as pd

import numpy as np

import math

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

import time

import warnings 

warnings.filterwarnings('ignore')
# load metadata

t1 = time.time()

topics = pd.read_csv('../input/trec-covid-information-retrieval/topics-rnd3.csv')

qrels = pd.read_csv('../input/trec-covid-information-retrieval/qrels.csv')

docids = pd.read_csv('../input/trec-covid-information-retrieval/docids-rnd3.txt')



t2 = time.time()

print('Elapsed time:', t2-t1)
topics.head()
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

topics.groupby('query').mean().sort_values(by='topic-id', ascending=False)['topic-id'].plot(kind='bar', color='r',width=0.3,title='topic-id : query', fontsize=8)

plt.xticks(rotation = 90)

plt.ylabel('query')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(topics.groupby('query').mean().sort_values(by='topic-id', ascending=False)['topic-id'][[1,2]])

print(topics.groupby('query').mean().sort_values(by='topic-id', ascending=False)['topic-id'][[4,5,6]])
topics.question.value_counts()[0:50].plot(kind='bar')

plt.grid()

plt.show()
topics.narrative.value_counts()[0:100].plot(kind='bar')

plt.grid()

plt.show()
qrels.head()
docids.head()
import re

text=open("../input/trec-covid-information-retrieval/docids-rnd3.txt", encoding='ISO-8859-1')

text=text.read()

print(text)
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

qrels.groupby('cord-id').mean().sort_values(by='topic-id', ascending=False)['topic-id'].plot(kind='bar', color='r',width=0.3,title='topic-id : cord-id', fontsize=8)

plt.xticks(rotation = 90)

plt.ylabel('cord-id')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(qrels.groupby('cord-id').mean().sort_values(by='topic-id', ascending=False)['topic-id'][[1,2]])

print(qrels.groupby('cord-id').mean().sort_values(by='topic-id', ascending=False)['topic-id'][[4,5,6]])
qrels_sub = qrels[['topic-id','cord-id']]

qrels_sub.head(100)
#Create a  DataFrame

submission = pd.DataFrame({'topic-id':qrels_sub['topic-id'],'cord-id':qrels_sub['cord-id']})

                        



#Visualize the first 10 rows

submission.head(10)
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)