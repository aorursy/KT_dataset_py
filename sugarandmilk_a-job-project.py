# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.figure_factory import create_gantt
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import os, re, gc 

from io import StringIO
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

filename = '/kaggle/input/corpus-of-russian-news-articles-from-lenta/lenta-ru-news.csv'
df = pd.read_csv(filename)
df.head(10)
df.info()
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.floor('D')
countNewsPerYear = df.groupby(df['date'].dt.year).size()
plt.figure(figsize=(12, 8))
fig = countNewsPerYear.plot(kind='bar')
fig.set_title('TOTAL NUMBER OF NEWS ARTICLES')
fig.set_xlabel('YEAR')
fig.set_ylabel('NUMBER OF NEWS ARTICLES')
df.loc[df['date'].dt.year == 1914]
df['date'][:5] = df['date'][:5] + pd.offsets.DateOffset(year=2014)

df = df.sort_values(['date'], ascending=True)
df = df.reset_index(drop=True)
amountOfNans = df.iloc[:, 2:4].isna().sum() 
amountOfNans.sort_values(ascending = False ) / df.shape[0] * 100
df = df[df['topic'].notna()]
df = df[df['text'].notna()]

#refresh the indexes
df = df.set_index(np.arange(len(df.index)))
#Remove extra space from 'Культпросвет '
df_topic = []

for i in df['topic']:
    if i == 'Культпросвет ':
        df_topic.append('Культпросвет')
    else:
        df_topic.append(i)

df['topic'] = df_topic

del df_topic
gc.collect() 
# Unique topic names
nameOfTopics = df['topic'].unique()

df_dict = []
for i in nameOfTopics:      
    serie = df[df['topic'] == i]   
    # add first, last date appearance for each topic
    df_dict.append(dict(Task=i, Start=serie.iloc[0, 5], Finish=serie.iloc[-1, 5]))
    
fig = create_gantt(df_dict, title='The date appearance of topics', height=600, bar_width=0.5, width=600)
fig.show()

del df_dict
gc.collect() 
countNewsPerTopic = df.groupby(df['topic']).size()
countNewsPerTopic = countNewsPerTopic.sort_values(ascending = False)

plt.figure(figsize=(12, 8))
fig = countNewsPerTopic.plot(kind='bar')
fig.set_title('TOTAL NUMBER OF NEWS ARTICLES')
fig.set_xlabel('TOPIC')
fig.set_ylabel('NUMBER OF NEWS ARTICLES')
rareTopics = ['Крым','Культпросвет', 'Легпром', 'Библиотека', 'Оружие', 'ЧМ-2014', 'Сочи', 'МедНовости', '69-я параллель'] 
percOfRareTopic = sum((df['topic'].isin(rareTopics)))/ len(df['topic']) * 100

print(f'{percOfRareTopic:.3f}% for rare topics')
mask = np.logical_not(df['topic'].isin(rareTopics))
df = df[mask]


del mask
gc.collect() 
#refresh the indexes
df = df.set_index(np.arange(len(df.index)))
#Remove url and date from the dataset
df = df.drop(['url', 'date'], axis = 1)
df.to_csv('textFromEDA.csv', index = False)



preprocFile = '/kaggle/input/a-job-project/preprocess_text2.csv'

new_df = pd.read_csv(preprocFile)