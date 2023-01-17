# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd

import numpy as np

import re

import string

import nltk

import datetime as dt

import seaborn as sns

import networkx as nx

import matplotlib.pyplot as plt

from gensim.models import word2vec



from sklearn.manifold import TSNE

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#fields1 = ['Last Name','First Name','Date of Birth','Gender','Race','Date Received','County','Date of Offense']

#fields2 = ['Last Name','First Name','Age','Date','Race','County','Last Statement']

#Death_Row = pd.read_csv("../input/death-row.csv")

#offenders = pd.read_csv("../input/offenders.csv")





Death_Row = pd.read_csv("../input/death-row.csv")

non_us = pd.read_csv("../input/non-us-offenders.csv")

non_on = pd.read_csv("../input/not-on-death-row.csv")

offenders = pd.read_csv("../input/offenders.csv",encoding = "latin1")
DR_Race = Death_Row['Race'].value_counts()

sns.barplot(DR_Race.index,DR_Race.values,color="b")
DR_Gender = Death_Row['Gender'].value_counts()

sns.barplot(DR_Gender.index,DR_Gender.values,color="b")
DR_County = Death_Row['County'].value_counts()

plt.figure(figsize=(10,4))

plt.xticks(rotation = 'vertical')

sns.barplot(DR_County.index,DR_County.values,alpha=0.8)
DR_County
Death_Row['Date of Birth'] = pd.to_datetime(Death_Row['Date of Birth'])

Death_Row['day'] = pd.DatetimeIndex(Death_Row['Date of Birth']).day 

Death_Row['month'] = pd.DatetimeIndex(Death_Row['Date of Birth']).month

Death_Row['year'] = pd.DatetimeIndex(Death_Row['Date of Birth']).year





YM= Death_Row['year'].max()

YMI = Death_Row['year'].min()



print('Maximum Year', YM)

print ('Minimum Year', YMI)



Dobd = Death_Row['day'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Dobd.index, Dobd.values, alpha=0.8)

plt.xlabel("Interrlation in Birth Date")

plt.show()



Dobm = Death_Row['month'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Dobm.index, Dobm.values, alpha=0.8)

plt.xlabel("Interrlation in Birth Month")

plt.show()



Doby= Death_Row['year'].value_counts()

plt.figure(figsize=(13,6))

sns.barplot(Doby.index, Doby.values, alpha=0.8)

plt.xticks(rotation = 'vertical')

plt.xlabel("Interrlation in Year Of Birth")

plt.show()
Death_Row['Date of Offense'] = pd.to_datetime(Death_Row['Date of Offense'])

Death_Row['Date Received'] = pd.to_datetime(Death_Row['Date Received'])

Death_Row['Timefor_Justice'] = Death_Row['Date Received'] - Death_Row['Date of Offense']
Death_Row['Timefor_Justice'].head(10)
Death_Row['Timefor_Justice'].max()
Death_Row['Timefor_Justice'].min()
offenders.head(1)
#sns.distplot(offenders['Age'])
Graphx= nx.Graph()

Graphx = nx.from_pandas_dataframe(offenders,source='Race',target='County')

print (nx.info(Graphx))

plt.figure(figsize=(10,8)) 

nx.draw(Graphx,alpha=0.5,with_labels=True,node_size=15,node_color='#FF0000')

plt.show()
Graphx= nx.Graph()

Graphx = nx.from_pandas_dataframe(offenders,source='Race',target='Age')

print (nx.info(Graphx))

plt.figure(figsize=(9,10)) 

nx.draw(Graphx,alpha=0.10, node_color="blue",with_labels=True,node_size=50)

plt.show()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.sentiment.util import*

senti = SentimentIntensityAnalyzer()

#%%time



offenders['Senti_Compound_Score'] = offenders['Last Statement'].apply(lambda x : senti.polarity_scores(x)['compound'])

offenders['Neutral_score'] = offenders['Last Statement'].apply(lambda x : senti.polarity_scores(x)['neu'])

offenders['Positive_score'] = offenders['Last Statement'].apply(lambda x : senti.polarity_scores(x)['pos'])

offenders['Negative_score'] = offenders['Last Statement'].apply(lambda x : senti.polarity_scores(x)['neg'])



offenders.loc[offenders.Senti_Compound_Score >0 ,'Overall_Sentiment']='Positive'

offenders.loc[offenders.Senti_Compound_Score == 0, 'Overall_Sentiment'] = 'Neutral'

offenders.loc[offenders.Senti_Compound_Score < 0,'Overall_Sentiment'] = 'Negative'
plt.figure(figsize=(10,5)) 

offenders.Overall_Sentiment.value_counts().plot(kind='bar',title="Last Statement")
%%timeit

def cleaning(s):

    s = str(s)

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    return s

offenders['Last Statement'] = [cleaning(s) for s in offenders['Last Statement']]
from nltk.corpus import stopwords

stop = stopwords.words('english')

offenders["Last Statement"] = offenders["Last Statement"].str.lower().str.split()

offenders['Last Statement'] = offenders['Last Statement'].apply(lambda x: [item for item in x if item not in stop])
words = pd.Series(' '.join(offenders['Last Statement'].astype(str)).lower().split(" ")).value_counts()[:25]

words
words.plot()