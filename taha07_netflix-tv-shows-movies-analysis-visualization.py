# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas_profiling

import missingno as msno

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.io as pio

import seaborn as sns

df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

df.head()
df.info()
profile = pandas_profiling.ProfileReport(df)

profile
n = msno.bar(df,color='purple')
df.drop(["director","cast"],axis =1,inplace=True)

df.head()
df['country'].value_counts()
df['country'].replace(np.nan,"United States",inplace=True)
df['date_added'].value_counts()
netflix_date = df[['date_added']].replace(np.nan,'Not Added')

netflix_date["release_month"] = netflix_date['date_added'].apply(lambda x: x.lstrip().split(" ")[0])

netflix_date.head()
netflix_date["release_month"].value_counts()
netflix_date['release_month'].replace('Not', 0,inplace=True)

netflix_date["release_month"].value_counts()
netflix_date.drop("date_added",axis=1,inplace=True)

netflix_date.head()
netflix = pd.concat([df,netflix_date],axis=1)

netflix.head()
netflix.drop("date_added",axis=1,inplace=True)

netflix.head()
netflix["rating"].value_counts()
netflix["rating"].isnull().sum()
netflix["rating"].replace(np.nan,"TV-MA",inplace=True)

netflix.isnull().sum()
netflix.head()
sns.set()

sns.countplot(x="type",data=netflix)

plt.show()
plt.figure(figsize=(12,9))

sns.countplot(x="rating",data=netflix,order= netflix['rating'].value_counts().index[0:14])
sns.set()

plt.figure(figsize=(30,9))

sns.countplot(x="release_year",data= netflix,order = netflix['release_year'].value_counts().index[0:40])

plt.xticks(rotation=45)

plt.show()
sns.set()

plt.figure(figsize=(20,8))

sns.countplot(x="release_month",data= netflix,order = netflix['release_month'].value_counts().index[0:12])

plt.xticks(rotation=45)

plt.show()
sns.set()

plt.figure(figsize=(25,9))

sns.countplot(x="rating",data= netflix,hue= "type",order = netflix['rating'].value_counts().index[0:15])

plt.xticks(rotation=45)

plt.show()
netflix["country"].value_counts().head()
sns.set()

plt.figure(figsize=(25,9))

sns.countplot(x="country",data= netflix,hue= "type",order = netflix['country'].value_counts().index[0:15])

plt.xticks(rotation=45)

plt.show()
top = netflix['country'].value_counts()[0:8]

top.index
fig = px.pie(netflix,values = top,names = top.index,labels= top.index)

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()
from wordcloud import WordCloud

wordcloud = WordCloud(background_color = "black",width=1730,height=970).generate(" ".join(netflix.country))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.axis("off")

plt.figure(1,figsize=(12,12))

plt.show()
wordcloud = WordCloud(background_color = "white",width=1730,height=970).generate(" ".join(netflix.title))

plt.figure(figsize=(15,10))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.axis("off")

plt.figure(1,figsize=(12,12))

plt.show()
top_listed_in=netflix["listed_in"].value_counts()[0:25]

top_listed_in.head()
sns.set()

plt.figure(figsize=(30,15))

sns.countplot(x='listed_in',data = netflix,order =netflix["listed_in"].value_counts().index[0:25])

plt.xticks(rotation = 90)

plt.show()
fig = px.pie(netflix,values = top_listed_in,names = top_listed_in.index,labels= top_listed_in.index)

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()
sns.set()

plt.figure(figsize=(30,15))

sns.countplot(x='listed_in',hue='rating',data = netflix,order =netflix["listed_in"].value_counts().index[0:10])

plt.xticks(rotation = 30)

plt.show()
old = netflix.sort_values("release_year",ascending=True)

old[["title","type","country","release_year"]].head(20)
kids_show=netflix[netflix["listed_in"] == "Kids' TV"].reset_index()

kids_show[["title","country","release_year"]].head(10)
netflix[netflix["country"] == "Bangladesh"]
Country = pd.DataFrame(netflix["country"].value_counts().reset_index().values,columns=["country","TotalShows"])

Country.head()
fig = px.choropleth(   

    locationmode='country names',

    locations=Country.country,

    featureidkey="Country.country",

    labels=Country["TotalShows"]

)

fig.show()