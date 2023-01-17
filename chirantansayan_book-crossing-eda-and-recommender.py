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
import seaborn as sns

import matplotlib.pyplot as plt
#Users

u_cols = ['user_id', 'location', 'age']

users = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1',low_memory=False)



#Books

i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']

items = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';', names=i_cols, encoding='latin-1',low_memory=False)



#Ratings

r_cols = ['user_id', 'isbn', 'rating']

ratings = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)
users.head(5)

users.shape
items.head(5)

items.shape
ratings.head(5)

ratings.shape
users = users.drop(users.index[0])

items = items.drop(items.index[0])

ratings = ratings.drop(ratings.index[0])
df = pd.merge(users, ratings, on='user_id')

df = pd.merge(df, items, on='isbn')

df.head(5)

df.shape
df = df[:102000]

df.shape
df = df.dropna()

df.drop(['img_s','img_m','img_l'],axis=1,inplace=True)

df.head()
df.describe()
df['age'] = df['age'].astype(int)

df['user_id'] = df['user_id'].astype(int)

df['rating'] = df['rating'].astype(int)
location = df.location.str.split(', ', n=2, expand=True)

location.columns=['city', 'state', 'country']



df['city'] = location['city']

df['state'] = location['state']

df['country'] = location['country']
df.head(5)
df.isna().sum()
df.groupby('book_title')['rating'].mean().sort_values(ascending=False).head(5)
df.groupby('book_title')['rating'].count().sort_values(ascending=False).head(5)
ratings = pd.DataFrame(df.groupby('book_title')['rating'].mean())

ratings.head(5)
ratings['num of ratings'] = pd.DataFrame(df.groupby('book_title')['rating'].count())

ratings.head(5)
ratings['num of ratings'].hist(bins=80)
ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
df.head(10)
df1 = df.pivot_table(index='user_id',columns='book_title',values='rating').fillna(0)

df1.head(10)
ratings.sort_values('num of ratings',ascending=False).head(10)
DV_rating = df1['The Da Vinci Code']

LP_rating = df1['Life of Pi']

HP_rating = df1['Harry Potter and the Goblet of Fire (Book 4)']

DV_rating.head()
similar_to_DV = df1.corrwith(DV_rating)

similar_to_LP = df1.corrwith(LP_rating)

similar_to_HP = df1.corrwith(HP_rating)

similar_to_DV.sort_values(ascending=False).head()
corr_DV = pd.DataFrame(similar_to_DV,columns=['correlation'])

corr_DV.dropna(inplace=True)

corr_DV = corr_DV.join(ratings['num of ratings'])

corr_DV.head()
corr_DV[corr_DV['num of ratings']>200].sort_values('correlation',ascending=False).head()
corr_LP = pd.DataFrame(similar_to_LP,columns=['correlation'])

corr_LP.dropna(inplace=True)

corr_LP = corr_LP.join(ratings['num of ratings'])
corr_LP[corr_LP['num of ratings']>200].sort_values('correlation',ascending=False).head()
corr_HP = pd.DataFrame(similar_to_HP,columns=['correlation'])

corr_HP.dropna(inplace=True)

corr_HP = corr_HP.join(ratings['num of ratings'])
corr_HP[corr_HP['num of ratings']>200].sort_values('correlation',ascending=False).head()