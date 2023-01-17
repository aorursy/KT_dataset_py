import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import wordcloud as wc

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style='dark')

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

df.head(10)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df.shape
df.isnull().sum()
columns = [' Rocket','Unnamed: 0','Unnamed: 0.1']

df.drop(columns, inplace=True, axis=1)
df.shape
#Converting Datum to DateTime datatype

df['DateTime'] = pd.to_datetime(df['Datum'])

#Extracting Year from Datum

df['Year']=df['DateTime'].apply(lambda x: x.year)

#Extracting Country from Location and trimming space before teh country name

df['Country']=df['Location'].apply(lambda x:x.split(',')[-1])

df['Country']=df['Country'].str.strip()

#Dropping the Datum column as it is redundant

df = df.drop('Datum',1)
df.head(10)
df['Country'].value_counts().head(10).plot.pie(autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2,figsize=(15,15),title='% Distribution Country-wise')
df.head()
df.groupby(['Company Name','Status Rocket'])['Status Rocket'].count().unstack().plot(kind='bar', stacked=True, figsize=(16,16),title='Rocket Status Company-wise')
grouped = df.groupby(['Country','Status Mission'])['Country'].count().unstack()

grouped.sort_index(ascending=False)

grouped.plot(kind='bar', stacked=True, figsize=(16,16),title='Mission Status Country-wise')
from wordcloud import WordCloud, STOPWORDS



df2 = df.query('Year > 2010')

company = " ".join(df2['Company Name'])

country = " ".join(df2['Country'])

#Defining a function to plot wordclouds

def plot_cloud(wordcloud):

    # Set figure size

    plt.figure(figsize=(30, 20))

    # Display image

    plt.imshow(wordcloud) 

    # No axis details

    plt.axis("off");

    

wordcloud = WordCloud(width = 1600, height = 1200, random_state=1, background_color='black', colormap='Pastel1', collocations=False,

                      stopwords = STOPWORDS).generate(company)

    

    

plot_cloud(wordcloud)
wordcloud = WordCloud(width = 1600, height = 1200, random_state=1, background_color='black', colormap='Pastel1', collocations=False,

                      stopwords = STOPWORDS).generate(country)

    

    

plot_cloud(wordcloud)