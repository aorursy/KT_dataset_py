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
movies = pd.read_csv('/kaggle/input/movietweetings/movies.dat', delimiter='::', engine='python', header=None, names = ['Movie ID', 'Movie Title', 'Genre'])

users = pd.read_csv('/kaggle/input/movietweetings/users.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Twitter ID'])

ratings = pd.read_csv('/kaggle/input/movietweetings/ratings.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Movie ID', 'Rating', 'Rating Timestamp'])

pd.set_option('display.max_columns', None)



movies.head()
movies.iloc[3]['Movie Title']
ratings.head()
movies.dropna(how='any',axis=0,inplace=True)
movies.isnull().sum()
movies.head()
p=movies['Movie Title'].str.split('(',expand=True)[1]

movies['year']=p.str.split(")",expand=True)[0]

movies
movies['Movie Title']=movies['Movie Title'].str.split('(',expand=True)[0]



movies.head(10)
p=movies['Genre'].str.split('|',expand=True)

movies['Genre']=p[0]
movies['Genre'].loc[movies['Genre']=='Short']=p[1]

movies.head(10)

        
movies.isnull().sum()
movies['Genre'].unique()
df=pd.merge(movies,ratings,how='right')

df.head(10)
df=df.drop('Rating Timestamp',axis=1)

df.loc[df['Movie Title']==np.nan]
df
df.isnull().sum()
df.dropna(how='any',axis=0,inplace=True)
df.head(2)
df[['Movie Title','Rating']].groupby(['Movie Title']).agg({'Rating':np.mean}).sort_values(by='Rating',ascending=False).head(10)

df[['Movie Title','Genre','Rating','year']].groupby(['year','Movie Title']).agg({'Rating':np.mean})
df.rename(columns={'Rating':'raw_ratings'})

df.head(2)
df['raw_ratings']=df['Rating']

df.head(2)


from surprise import KNNWithMeans

from surprise import Dataset

from surprise import accuracy

from surprise import Reader

from surprise.model_selection import train_test_split
reader=Reader(rating_scale=(1,10))

data=Dataset.load_from_df(df[['User ID','Movie ID','Rating']],reader)
[trainset,testset]=train_test_split(data,test_size=.4,shuffle=True)
#recom=KNNWithMeans(k=17,sim_options={'name':'cosine','user_based':True})

#recom.fit(data.build_full_trainset()) #using full tarining set and not doning

#test_pred=recom.test(testset)

#RMSE=accuracy.rmse(test_pred)