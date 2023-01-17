# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading the dataset

data=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
data.shape
data.columns
data.head()
mov=data.query("type=='Movie'")

mov['min']=mov['duration'].str.split(' ',expand=True)[0]

mov['min']=mov['min'].astype(int)

mov['hr']=mov['min']/60
top20run=mov.sort_values(by='hr',ascending=False).head(20)

plt.figure(figsize=(10,7))

sns.barplot(data=top20run,y='title',x='hr',hue='country',dodge=False)

plt.legend(loc='lower right')

plt.title('Top 10 movies by Run Time')

plt.xlabel('Hours')

plt.ylabel('Movie name')

plt.show()
tv=data.query("type=='TV Show'")

tv['sea']=tv['duration'].str.split(' ',expand=True)[0]

tv['sea']=tv['sea'].astype(int)
top20tv=tv.sort_values(by='sea',ascending=False).head(20)

plt.figure(figsize=(10,7))

sns.barplot(data=top20tv,y='title',x='sea',hue='country',dodge=False)

plt.legend(loc='lower right')

plt.title('Top TV show by Run Time')

plt.xlabel('Seasons')

plt.ylabel('Movie name')

plt.show()
sns.set(style="darkgrid", palette="pastel", color_codes=True)

plt.figure(figsize=(5,10))

sns.countplot(y='director',data=data,order = data['director'].value_counts().head(20).index)

plt.show()
import cufflinks as cf

import plotly.offline

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

indcast=[]

ind=data.query('country=="India"')

for i in ind['cast']:

    indcast.append(i)

newls=[]

for i in indcast:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

ind_df['name'].value_counts().head(20).plot(kind="bar",width=1,color='red',edgecolor='lightgreen',figsize=(10,8))

plt.xlabel("ACTORS")

plt.ylabel("No. Of Movies")

plt.title("Indian stars with max movies on netflix")

plt.show()
us=data[data['country'].str.contains('United States',na=False)]

uscast=[]

for i in us['cast']:

    uscast.append(i)

newls1=[]

for i in uscast:

    newls1.append(str(i).split(',')[0])

    

usdf=pd.DataFrame(newls1,columns=['name'])

us_df=usdf.drop(usdf.query('name=="nan"').index)

us_df['name'].value_counts().head(20).plot(kind="bar",width=.8,edgecolor='black',figsize=(10,8))

plt.xlabel("ACTORS")

plt.ylabel("No. Of Movies")

plt.title("US stars with max movies on netflix")

plt.show()
rat=data.groupby('rating')[['show_id']].count().reset_index()

plt.figure(figsize=(10,7))

plt.pie(rat['show_id'],autopct='%1.1f%%', startangle=90, pctdistance=1.2,shadow=True)

centre_circle = plt.Circle((0,0),0.4,fc='white')

fig=plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')

plt.legend(data['rating'])

plt.show()
data['date']=pd.to_datetime(data['date_added'])

data['month']=data['date'].dt.strftime('%b')

data.groupby('month')[['rating']].count().reset_index().sort_values(by='rating',ascending=False).plot(kind='bar',\

                                                                               width=1,color='green',figsize=(10,8))

plt.xlabel("Months")

plt.ylabel("Count")

plt.show()
!pip install rake_nltk

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer

from rake_nltk import Rake
new_df = data[['title','director','cast','listed_in','description']]

new_df.head()
new_df['director']=new_df['director'].fillna(' ')

new_df['director']=new_df['director'].astype('str')
new_df['cast']=new_df['cast'].fillna(' ')

new_df['cast']=new_df['cast'].astype('str')
new_df['bag_of_word']= ''

for index,row in new_df.iterrows():

    plot = row['description']

    r = Rake()

    r.extract_keywords_from_text(plot)

    keyword_score = r.get_word_degrees()

    g = ''.join(row['listed_in'].split(',')).lower()

    d = ''.join(row['director'].replace(' ','').split(',')).lower()

    a = ' '.join(row['cast'].replace(' ','').split(',')).lower()

    k = ' '.join(list(keyword_score.keys()))

    row['bag_of_word'] = g + ' ' + ' ' + d + ' ' + a + ' ' + k



mydf = new_df[['title','bag_of_word']]

mydf.head()
c = CountVectorizer()

count_mat = c.fit_transform(mydf['bag_of_word'])

cosine_sim = cosine_similarity(count_mat,count_mat)

print(cosine_sim)
indices = pd.Series(mydf['title'])

def recommend_movie(name):

    movie=[]

    idx = indices[indices == name].index[0]

    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10 = sort_index.iloc[1:11]

    for i in top_10.index:

        movie.append(indices[i])

    return movie
recommend_movie('Transformers Prime')
recommend_movie('Oh! Baby (Malayalam)')
recommend_movie('Naruto Shippuden : Blood Prison')
recommend_movie('Main Hoon Na')