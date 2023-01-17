import numpy as np 

import pandas as pd

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from wordcloud import WordCloud

from collections import Counter

%matplotlib inline

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#load data

data=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

data.head()
data.tail()
data.info()
data.shape
data.columns
data.describe().T
data.isnull().values.any()
data.isnull().sum()
data.drop(["director","cast"],axis=1,inplace=True)

data.head()
data.country.value_counts()
data.country.replace(np.nan,"United States",inplace=True)
data.date_added.value_counts()
df = data[['date_added']].replace(np.nan,'Not Date')

df["release_month"] = df['date_added'].apply(lambda x: x.lstrip().split(" ")[0])

df.head()



df.release_month.value_counts()
df.release_month.replace("Not",0,inplace=True)

df.release_month.value_counts()
df.drop("date_added",axis=1,inplace=True)

df
new_data=pd.concat([data,df],axis=1)

new_data.head()
new_data.drop("date_added",axis=1,inplace=True)

new_data.head()
new_data.isnull().sum()
new_data.rating.value_counts()
new_data.rating.replace(np.nan,"TV-MA",inplace=True)

new_data.isnull().sum()
new_data.head()
#Movie or Tv Show

sns.countplot(x="type",data=new_data)

new_data.type.value_counts()
plt.figure(figsize=(10,7))

sns.countplot(x=new_data.country,order=new_data.country.value_counts().index[0:10]);

plt.xticks(rotation=45)

plt.show()
new_data.head()
plt.figure(figsize=(30,10))

sns.countplot(x=new_data.release_year,order=new_data.release_year.value_counts().index[0:30]);

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x=new_data.rating,order=new_data.rating.value_counts().index[0:20]);

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x=new_data.release_month,order=new_data.release_month.value_counts().index[0:12]);

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(x=new_data.country,hue= new_data.type,order = new_data['country'].value_counts().index[0:17])

plt.xticks(rotation=45)

plt.show()
labels = new_data.country.value_counts()[0:6].index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = new_data.country.value_counts()[0:6].values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x=new_data.rating,hue=new_data.type,order=new_data.rating.value_counts().index[0:20]);

plt.xticks(rotation=45)

plt.show()