import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
netflix=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
netflix.head(10)
netflix.shape
netflix.columns
netflix.isnull().sum()
netflix.nunique()
netflix.duplicated().sum()
df = netflix.copy()
df.shape
df=df.dropna()

df.shape
df.head(10)
df["date_added"] = pd.to_datetime(df['date_added'])

df['day_added'] = df['date_added'].dt.day

df['year_added'] = df['date_added'].dt.year

df['month_added']=df['date_added'].dt.month

df['year_added'].astype(int);

df['day_added'].astype(int);
df.head(10)
sns.countplot(netflix['type'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Type')
sns.countplot(netflix['rating'])

sns.countplot(netflix['rating']).set_xticklabels(sns.countplot(netflix['rating']).get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(13,13)

plt.title('Rating')
plt.figure(figsize=(10,8))

sns.countplot(x='rating',hue='type',data=netflix)

plt.title('Relation between Type and Rating')

plt.show()
labels = ['Movie', 'TV show']

size = netflix['type'].value_counts()

colors = plt.cm.Wistia(np.linspace(0, 1, 2))

explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size,labels=labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Distribution of Type', fontsize = 25)

plt.legend()

plt.show()
netflix['rating'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(10,8))

plt.show()
from wordcloud import WordCloud
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.country))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('country.png')

plt.show()
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.cast))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.director))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('director.png')

plt.show()
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.listed_in))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('category.png')

plt.show()