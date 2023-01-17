import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

plt.style.use('seaborn-whitegrid')

sns.set_style('whitegrid')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')

print('Done!')
def data_inv(df):

    print('netflix movies and shows: ',df.shape[0])

    print('dataset variables: ',df.shape[1])

    print('-'*10)

    print('dateset columns: \n')

    print(df.columns)

    print('-'*10)

    print('data-type of each column: \n')

    print(df.dtypes)

    print('-'*10)

    print('missing rows in each column: \n')

    c=df.isnull().sum()

    print(c[c>0])

data_inv(df)
dups=df.duplicated(['title','country','type','release_year'])

df[dups]
df=df.drop_duplicates(['title','country','type','release_year'])
df=df.drop('show_id',axis=1)
df['cast']=df['cast'].replace(np.nan,'Unknown')

def cast_counter(cast):

    if cast=='Unknown':

        return 0

    else:

        lst=cast.split(', ')

        length=len(lst)

        return length

df['number_of_cast']=df['cast'].apply(cast_counter)

df['cast']=df['cast'].replace('Unknown',np.nan)
df=df.reset_index()
df['rating']=df['rating'].fillna(df['rating'].mode()[0])
df['date_added']=df['date_added'].fillna('January 1, {}'.format(str(df['release_year'].mode()[0])))
for i,j in zip(df['country'].values,df.index):

    if i==np.nan:

        if ('Anime' in df.loc[j,'listed_in']) or ('anime' in df.loc[j,'listed_in']):

                df.loc[j,'country']='Japan'

        else:

            continue

    else:

        continue
import re

months={

    'January':1,

    'February':2,

    'March':3,

    'April':4,

    'May':5,

    'June':6,

    'July':7,

    'August':8,

    'September':9,

    'October':10,

    'November':11,

    'December':12

}

date_lst=[]

for i in df['date_added'].values:

    str1=re.findall('([a-zA-Z]+)\s[0-9]+\,\s[0-9]+',i)

    str2=re.findall('[a-zA-Z]+\s([0-9]+)\,\s[0-9]+',i)

    str3=re.findall('[a-zA-Z]+\s[0-9]+\,\s([0-9]+)',i)

    date='{}-{}-{}'.format(str3[0],months[str1[0]],str2[0])

    date_lst.append(date)
df['date_added_cleaned']=date_lst
df=df.drop('date_added',axis=1)
df['date_added_cleaned']=df['date_added_cleaned'].astype('datetime64[ns]')
for i in df.index:

    if df.loc[i,'rating']=='UR':

        df.loc[i,'rating']='NR'
plt.figure(figsize=(8,6))

df['rating'].value_counts(normalize=True).plot.bar()

plt.title('Distribution of rating categories')

plt.xlabel('rating')

plt.ylabel('relative frequency')

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(x='rating',hue='type',data=df)

plt.title('comparing frequency between type and rating')

plt.show()
df['country'].value_counts().sort_values(ascending=False)
top_productive_countries=df[(df['country']=='United States')|(df['country']=='India')|(df['country']=='United Kingdom')|(df['country']=='Japan')|

                             (df['country']=='Canada')|(df['country']=='Spain')]

plt.figure(figsize=(10,8))

sns.countplot(x='country',hue='type',data=top_productive_countries)

plt.title('comparing between the types that the top countries produce')

plt.show()
for i in top_productive_countries['country'].unique():

    print(i)

    print(top_productive_countries[top_productive_countries['country']==i]['rating'].value_counts(normalize=True)*100)

    print('-'*10)
df['year_added']=df['date_added_cleaned'].dt.year
df['type'].value_counts(normalize=True)
df.groupby('year_added')['type'].value_counts(normalize=True)*100
dups=df.duplicated(['title'])

df[dups]['title']
for i in df[dups]['title'].values:

    print(df[df['title']==i][['title','type','release_year','country']])

    print('-'*40)
plt.figure(figsize=(10,8))

df['year_added'].value_counts().plot.bar()

plt.title('distribution of year-added')

plt.ylabel('relative frequency')

plt.xlabel('year_added')

plt.show()
counts=0

for i,j in zip(df['release_year'].values,df['year_added'].values):

    if i!=j:

        counts+=1

print('number of contents that its release year differ from the year added to netflix are ',str(counts))