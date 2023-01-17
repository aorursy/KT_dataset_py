import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

from scipy.stats import pearsonr

%matplotlib inline 

import os

print(os.listdir("../input"))
top2018= pd.read_csv('../input/top2018.csv')

top2018.shape
top2018.head()
def convertMillis(millis):

     minutes=(millis/(1000*60))%60

     return minutes

top2018['duration_min']=convertMillis(top2018['duration_ms'])

top2018.drop('duration_ms', axis=1, inplace=True)
sns.heatmap(top2018.corr(), cmap= 'CMRmap')
top2018['artists'].value_counts().head(20)
sns.set_style(style='dark')

sns.distplot(top2018['danceability'],hist=True,kde=True)
sns.boxplot(data=top2018['danceability'])
top25= top2018.iloc[0:25, :]

sns.distplot(top2018['valence'])
happy_songs= pd.DataFrame(top25[top25['valence']>0.5])

happy_songs
sns.heatmap(happy_songs.corr(), annot=True)
top2018[['name','artists','danceability','energy','valence','tempo']].sort_values(by='valence',ascending=False).head(10)
live_songs = top2018[top2018['liveness']>0.6]

live_songs