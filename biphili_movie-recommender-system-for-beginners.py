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
import matplotlib.pyplot as plt
md = pd. read_csv('../input/the-movies-dataset/movies_metadata.csv')

md.head()
md.shape
md = md[md['vote_count'] >= 50]  

md.head()
md.shape


#md=md.iloc[:22000]
md.columns
md.drop(['adult','belongs_to_collection','budget','genres','homepage','original_language','original_title','overview','popularity','poster_path','production_companies','release_date','revenue','runtime','production_countries','spoken_languages','status','tagline','video'],axis=1, inplace=True)

md.head()
md.describe()
#md.groupby('title').describe()
md.groupby('title')['vote_average'].describe()
md1=md.drop(['id','imdb_id'],axis=1)

md1.head()
md1.shape
md1['vote_average'].plot(bins=100,kind='hist',color='red')

plt.ioff()
md1['vote_count'].plot(bins=80,kind='hist',color='red')

plt.xlim(xmin=0, xmax = 2000)

plt.ioff()
md1[md1['vote_average']==8]
md1.sort_values('vote_count',ascending=False).head(10)
md1.sort_values('vote_count',ascending=True).head(200)
md.info()
md.head()
userid_movietitle_matrix=md.pivot_table(index='imdb_id',columns='title',values='vote_average')

userid_movietitle_matrix
Inception=userid_movietitle_matrix['Inception']

Inception
Inception_correlations=pd.DataFrame(userid_movietitle_matrix.corrwith(Inception),columns=['Correlation'])

Inception_correlations