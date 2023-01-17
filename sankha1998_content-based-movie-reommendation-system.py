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
movie=pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')

rating=pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')
movie.shape
# movie and rating are sutable for analysis

movie_details=movie.merge(rating,on='movieId')
movie_details.head()
movie_details.drop(columns=['timestamp'],inplace=True)
total_ratings=movie_details.groupby(['movieId','genres']).sum()['rating'].reset_index()
df=movie_details.copy()
df.drop_duplicates(['title','genres'],inplace=True) 
df=df.merge(total_ratings,on='movieId')
df.drop(columns=['userId','rating_x','genres_y'],inplace=True)
df.rename(columns={'genres_x':'genres','rating_y':'rating'},inplace=True)
df.head()
df['rating']=df['rating'].astype(int)
df.dtypes
df = df[df['rating']>100]
df.shape
df['genres'].value_counts()
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=1)

x = tfv.fit_transform(df['genres'])
from sklearn.metrics.pairwise import sigmoid_kernel

model = sigmoid_kernel(x, x)
df1=df.copy()

ti=[]

for i in df1['title']:

    ti.append(i.split(' (')[0])

df1['title']=ti
def recommendations(title):

    i_d=[]

    indices=pd.Series(df1.index,index=df1['title']).drop_duplicates()

    idx = indices[title]

    dis_scores = list(enumerate(model[idx]))

    dis_scores = sorted(dis_scores, key=lambda x: x[1], reverse=True)

    dis_scores = dis_scores[1:31]

    idn = [i[0] for i in dis_scores]

    final =df1.iloc[idn].reset_index()

    idn = [i for i in final['index']]

    for j in idn:

        if(j<15951):

            i_d.append(j)

    indices=pd.Series(df.index,index=df['title']).drop_duplicates()

    for i in range(1,8):

        if (idn):

            print(indices.iloc[i_d].index[i])

     

recommendations('Before and After')