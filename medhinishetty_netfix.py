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

        

data=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')



# Any results you write to the current directory are saved as output.
data.sample(4)
data.shape
df = data.copy()

df.shape

import seaborn as sns

import matplotlib.pyplot as plt




labels = ['Movie', 'TV show']

size = data['type'].value_counts()

plt.figure(figsize=(9,6))

plt.pie(size,labels=labels,autopct='%1.1f%%')

 
plt.figure(figsize=(12,6))

sns.countplot(data['rating'])
plt.figure(figsize=(12,6))

sns.countplot(x='rating',hue='type',data=data).set_title('rating ')
df2 = data.copy()

df2 = df2[df2['date_added'].notna()]

df2['date']=pd.to_datetime(df2['date_added'])

df2['month']=df2['date'].dt.strftime('%b')

df2['month'].value_counts()

plt.figure(figsize=(12,6))

sns.countplot(x='month',hue='type',data=df2).set_title('Movies and TV shows added')
df2['date']=pd.to_datetime(df2['date_added'])

df2['Year']=df2['date'].dt.strftime('%Y')

plt.figure(figsize=(12,6))

sns.countplot(x='Year',hue='type',data=df2).set_title('Movies and TV shows added')   

df3=data[data.release_year>1999]

plt.figure(figsize=(30,6))

sns.countplot('release_year',data=df3).set_title('release_year')
df5=data['director'].value_counts()

df5=df5[df5>3]

df5


plt.figure(figsize=(5,10))

sns.countplot(y='director',data=data,order = data['director'].value_counts().head(20).index).set_title('director')

plt.show()
plt.figure(figsize=(8,10))

sns.countplot(y='country',hue='type',data=data,order = data['country'].value_counts().head(15).index).set_title('Country wise distribution')

plt.show()
df6=data[data.country=='India']

plt.figure(figsize=(5,10))

sns.countplot(y='director',data=df6,order = df6['director'].value_counts().head(30).index).set_title(' Indian director')

plt.show()
sns.countplot(y='cast',data=data,order = data['cast'].value_counts().head(15).index) 

plt.figure(figsize=(5,10))

plt.show()
cast=[]

for i in data['cast']:

    cast.append(i)

newls=[]

for i in cast:

    newls.append(str(i).split(',')[0])

df7=pd.DataFrame(newls,columns=['name'])

df7=df7.drop(df7.query('name=="nan"').index)

plt.figure(figsize=(10,10))

sns.countplot(y='name',data=df7,order = df7['name'].value_counts().head(20).index)



plt.show()
df9=data[data.type=="Movie"]

indcast=[]

ind=df9.query('country=="India"')

for i in ind['cast']:

    indcast.append(i)

newls=[]

for i in indcast:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

plt.figure(figsize=(10,10))

sns.countplot(y='name',data=ind_df,order = ind_df['name'].value_counts().head(20).index)

plt.title("Indian stars with max movies on netflix")

plt.show()
df9=data[data.type=="Movie"]

usacast=[]

usa=df9.query('country=="United States"')

for i in usa['cast']:

    usacast.append(i)

newls=[]

for i in usacast:

    newls.append(str(i).split(',')[0])

usadf=pd.DataFrame(newls,columns=['name'])

usa_df=usadf.drop(usadf.query('name=="nan"').index)

plt.figure(figsize=(10,10))

sns.countplot(y='name',data=usa_df,order = usa_df['name'].value_counts().head(20).index)

plt.title("Usa stars with max movies on netflix")

plt.show()
df9=data[data.type=="TV Show"]

indcast=[]

ind=df9.query('country=="India"')

for i in ind['cast']:

    indcast.append(i)

newls=[]

for i in indcast:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

plt.figure(figsize=(10,10))

sns.countplot(y='name',data=ind_df,order = ind_df['name'].value_counts().head(20).index)

plt.title("Indian stars with max TV SHOW on netflix")

plt.show()


l1=[]

for i in data['listed_in']:

    l1.append(i)

newls=[]

for i in l1:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

plt.figure(figsize=(10,10))

sns.countplot(y='name',data=ind_df,order = ind_df['name'].value_counts().index)

plt.title("genre on netflix")

plt.show()
d=data.query('country=="United States"')

l1=[]

for i in d['listed_in']:

    l1.append(i)

newls=[]

for i in l1:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

plt.figure(figsize=(10,10))

sns.countplot(y='name',data=ind_df,order = ind_df['name'].value_counts().index)

plt.title("IN USA genre on netflix")

plt.show()
d=data.query('country=="India"')

l1=[]

for i in d['listed_in']:

    l1.append(i)

newls=[]

for i in l1:

    newls.append(str(i).split(',')[0])

inddf=pd.DataFrame(newls,columns=['name'])

ind_df=inddf.drop(inddf.query('name=="nan"').index)

plt.figure(figsize=(10,10))

sns.countplot(y='name',data=ind_df,order = ind_df['name'].value_counts().index)

plt.title("IN INDIA genre on netflix")

plt.show()
!pip install rake-nltk
from rake_nltk import Rake

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
data2 = data[['title','director','cast','listed_in','description']]

data2.head()
data2['director']=data2['director'].fillna(' ')

data2['director']=data2['director'].astype('str')

data2['cast']=data2['cast'].fillna(' ')

data2['cast']=data2['cast'].astype('str')
data2['Description']=""

for i ,r in data2.iterrows():

    rake=Rake()

    

    rake.extract_keywords_from_text(r['description'])

    score_keyword=rake.get_word_degrees()

    a=''.join(r['listed_in'].split(',')).lower()

    b=''.join(r['director'].replace(' ','').split(',')).lower()

    c=''.join(r['cast'].replace(' ','').split(',')).lower()

    k = ' '.join(list(score_keyword.keys()))

    r['Description']=a+' '+b+' '+c+' '+k

data3=data2[['title','Description']]
data3.sample(5)
count=CountVectorizer()

matrix=count.fit_transform(data3['Description'])

cosine_sim = cosine_similarity(matrix,matrix)

print(cosine_sim)
ind= pd.Series(data3['title'])

def recommend(name):

    movie=[]

    idx = ind[ind == name].index[0]

    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10 = sort_index.iloc[1:11]

    for i in top_10.index:

        movie.append(ind[i])

    return movie
recommend('3Below: Tales of Arcadia')
recommend('Khosla Ka Ghosla')
recommend('Dil Dhadakne Do')
recommend("Power Rangers Dino Thunder")