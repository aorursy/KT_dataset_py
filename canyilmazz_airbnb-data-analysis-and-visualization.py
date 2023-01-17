import numpy as np 

import pandas as pd 

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
data.info()
data.shape
data.isnull().sum()
#Drop unnecessary columns

data.drop(["id","host_name","last_review","name"],axis=1,inplace=True)

data.isnull().sum()
data.reviews_per_month.fillna(0,inplace=True)

data.isnull().sum()
data.head()
plt.figure(figsize=(10,7))

sns.countplot(x=data.neighbourhood_group,order=data.neighbourhood_group.value_counts().index[0:10],palette="plasma");

plt.title('Neighbourhood Group')

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x=data.neighbourhood,order=data.neighbourhood.value_counts().index[0:10],palette="plasma");

plt.xticks(rotation=45)

plt.title('Neighbourhood')

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x=data.room_type,order=data.room_type.value_counts().index);

plt.title('Room Type')

plt.show()
plt.figure(figsize=(10,7))

sns.boxplot(x="room_type",y="availability_365",data=data,palette='plasma')

plt.show()
plt.figure(figsize=(10,7))

sns.boxplot(x='neighbourhood_group',y='availability_365',data=data, palette='plasma')

plt.show()
plt.figure(figsize=(7,7))

sns.barplot(x="neighbourhood_group",y="price",hue="room_type",data=data)
plt.subplots(figsize=(30,20))

wordcloud = WordCloud(

                          background_color='black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(data.neighbourhood))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('neighbourhood.png')

plt.show()