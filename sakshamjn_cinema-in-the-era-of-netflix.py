# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

#plt.style.use('ggplot')
df=pd.read_csv('../input/netflix-shows/netflix_titles.csv')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n  :',df.columns.tolist())

print('\nMissing values :',df.isnull().values.sum())

print('\nUnique values  :  \n',df.nunique())
df.info()
df=df.dropna()
df["date_added"] = pd.to_datetime(df['date_added'])

df['day_added'] = df['date_added'].dt.day

df['year_added'] = df['date_added'].dt.year

df['month_added']=df['date_added'].dt.month

df['year_added'].astype(int);

df['day_added'].astype(int);

#df.year_added = df.year_added.astype(float)

#df.style.set_precision(0)

df.head()
print(df['type'].value_counts())
f,ax=plt.subplots(1,2,figsize=(18,8))

df['type'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Type of Movie')

ax[0].set_ylabel('Count')

sns.countplot('type',data=df,ax=ax[1],order=df['type'].value_counts().index)

ax[1].set_title('Count of Source')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

df['rating'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Movie Rating')

ax[0].set_ylabel('Count')

sns.countplot('rating',data=df,ax=ax[1],order=df['rating'].value_counts().index)

ax[1].set_title('Count of Rating')

plt.show()
group_country_movies=df.groupby('country')['show_id'].count().sort_values(ascending=False).head(10);

plt.subplots(figsize=(15,8));

group_country_movies.plot('bar',fontsize=12,color='blue');

plt.xlabel('Number of Movies',fontsize=12)

plt.ylabel('Country',fontsize=12)

plt.title('Movie count by Country',fontsize=12)

plt.ioff()
group_country_movies=df.groupby('year_added')['show_id'].count().sort_values(ascending=False).head(10);

plt.subplots(figsize=(15,8));

group_country_movies.plot('bar',fontsize=12,color='blue');

plt.xlabel('Number of Movies',fontsize=12)

plt.ylabel('Year',fontsize=12)

plt.title('Movie Count By Year',fontsize=12)

plt.ioff()
df['month_added'].value_counts();
ax=df.groupby('show_id')['month_added'].unique().value_counts().iloc[:-1].sort_index().plot('bar',color='blue',figsize=(15,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Number of Movies',fontsize=15)

ax.set_title('Number of Moves Based on Month',fontsize=15)

ax.set_xticklabels(('Jan','Feb','Mar','April','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'));

plt.show()



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

#df.head()
from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

plt.style.use('seaborn')

from PIL import Image

plt.style.use('seaborn')

wrds1 = df["title"].str.split("(").str[0].value_counts().keys()



wc1 = WordCloud(stopwords=STOPWORDS,scale=5,max_words=1000,colormap="rainbow",background_color="black").generate(" ".join(wrds1))

plt.figure(figsize=(20,14))

plt.imshow(wc1,interpolation="bilinear")

plt.axis("off")

plt.title("Key Words in Movie Titles",color='black',fontsize=20)

plt.show()
#df['description'][1]

df['length']=df['description'].str.len()

df.dropna();
plt.figure(figsize=(12,5))



g = sns.distplot(df['length'])

g.set_title("Price Distribuition Filtered 300", fontsize=20)

g.set_xlabel("Prices(US)", fontsize=15)

g.set_ylabel("Frequency Distribuition", fontsize=15)





plt.show()
plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="rating", y="length", data=df,width=0.8,linewidth=3)

ax.set_xlabel('Rating',fontsize=30)

ax.set_ylabel('Length of Description',fontsize=30)

plt.title('Length of Description Vs Rating',fontsize=40)

ax.tick_params(axis='x',labelsize=20,rotation=90)

ax.tick_params(axis='y',labelsize=20,rotation=0)

plt.grid()

plt.ioff()
import plotly.graph_objects as go

from collections import Counter

col = "listed_in"

categories = ", ".join(df['listed_in']).split(", ")

counter_list = Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

plt.figure(figsize=(12,5))

sns.barplot(values[0:20],labels[0:20]);

plt.xlabel('Count',fontsize=10)

#plt.ylabel('',fontsize=20)

plt.title('Movie Listing',fontsize=20)

#ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()