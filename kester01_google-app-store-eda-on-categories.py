# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

import plotly.graph_objects as go

import plotly.express as px

import datetime

from plotly.offline import init_notebook_mode, iplot

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)
df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

print(df.shape)

df.info()
df.dropna(subset=['Android Ver','Current Ver','Type','Content Rating'],inplace=True)
df_drop_null_rating = df.dropna(subset=['Rating'])
for category in df['Category'].unique():

    mask = df[df['Category']== category]['Rating'].isna()

    mask = mask[mask==True]

    rating_median = df_drop_null_rating[df_drop_null_rating['Category'] == category]['Rating'].median()

    df.at[mask.index,'Rating'] = rating_median
df.info()
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df.head()
pattern = re.compile('[+,M]')

df['Install No.'] = df['Installs'].apply(lambda x: pattern.sub('',x))

df['Install No.'] = df['Install No.'].astype(int)

df['Size'] = df['Size'].apply(lambda x: pattern.sub('',x))

df['Size']=pd.to_numeric(df['Size'],errors='coerce')
data=df.groupby('Category').agg({'Install No.':'sum','App': 'count'}).sort_values('Install No.', ascending=False).reset_index()
fig, ax1 = plt.subplots(figsize=(15,7))

plt.ticklabel_format(style='plain', axis='y')

g1 = sns.barplot(data=data,x='Category',y='Install No.', ci=None)

ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax2 = ax1.twinx()

g2 = sns.lineplot(data=data,x='Category',y='App', ci=None, linewidth = 2.5)

ax2.set(ylabel='Number of Apps')
fig, ax = plt.subplots(1,2,figsize=(20,7))

df.value_counts('Type').plot.pie(y='Type',startangle=90, explode=(0.2,0), title='Percentage of the Free App and Paid App', legend=False, autopct='%.2f', ax=ax[0])

ax[0].set(ylabel='Type of Apps')

df.groupby('Type').agg({'Install No.':sum}).plot.pie(y='Install No.', startangle=90, explode=(0.2,0), title='Percentage of Installs Number for Free App and Paid App', legend=False, autopct='%.2f', ax=ax[1])
g = sns.FacetGrid(data=df,  hue='Type', margin_titles=True,legend_out=False)

g.map(sns.distplot,'Rating',kde_kws={'bw':0.1}).add_legend()

g.fig.set_size_inches((10,7))
plt.figure(figsize=(25,7))

g = sns.boxenplot(data=df,x='Category',y='Rating')

g.set_xticklabels(g.get_xticklabels(),rotation=90)
g = df.groupby('Category').agg({'Rating':'std'}).sort_values('Rating',ascending=False).plot(kind='bar',figsize=(20,7),title='Standard Deviation of the Ratings in Different Category')

g.set(ylabel='Standard Deviation')
df['Install No.'] = df['Install No.'].astype(int)

df['Reviews'] = df['Reviews'].astype(int)

df_gb_insreviews = df.groupby('Category').agg({'Install No.':'sum','Reviews':'sum'})
df_gb_insreviews['Percentage of reviews(%)'] = (df_gb_insreviews['Reviews'] / df_gb_insreviews['Install No.']) * 100
plt.figure(figsize=(25,7))

g = df_gb_insreviews.sort_values('Percentage of reviews(%)', ascending=False)['Percentage of reviews(%)'].plot(kind='bar', title='Percentage of reviews in total install number')

g.set(ylabel='Percentage(%)')
df_reviews = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")

df_reviews.info()
df_reviews.dropna(subset=['Translated_Review','Sentiment','Sentiment_Polarity','Sentiment_Subjectivity'],how='any', inplace=True)

df_reviews.reset_index(inplace=True)

df_reviews.info()
df_reviews = df_reviews.merge(df[['Category','App']].drop_duplicates('App'),on=['App'],how ='left')
df_reviews.groupby('Category').agg({'Sentiment_Polarity':'mean','Sentiment_Subjectivity':'mean'}).sort_values(['Sentiment_Polarity','Sentiment_Subjectivity'],ascending=False).style.bar(subset=['Sentiment_Polarity', 'Sentiment_Subjectivity'], color='#d65f5f')