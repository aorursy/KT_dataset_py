# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import json

from sklearn.preprocessing import Imputer



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')



del credits['title']

df = pd.concat([movies, credits], axis=1)



newCols = ['id','title','release_date','popularity','vote_average','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status']



df2 = df[newCols]



df2.head()
json_columns = ['genres', 'keywords', 'production_countries',

                    'production_companies']

for column in json_columns:

    if df2[column][4800] == list:

        continue

    df2[column] = df2[column].apply(json.loads)
type(df2['genres'][6][0]['id'])
for y in range(len(df2['genres'])):

    for x in range(len(df2['genres'][y])):

        if ' ' in df2['genres'][y][x]['name']:

            df2['genres'][y][x]['name'] = df2['genres'][y][x]['name'].replace(' ','_')

            
df2
def pipe_flatten_names(d):

    return ' '.join([x['name'] for x in d])



df2['genres'] = df2['genres'].apply(pipe_flatten_names)

df_list = df2['genres'].tolist()
df_list
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()



vect.fit(df_list)



vect.get_feature_names()
df_list_dtm = vect.transform(df_list)

df_list_dtm.toarray()



df_genres = pd.DataFrame(df_list_dtm.toarray(), columns=vect.get_feature_names())



df_genres
for i in range(len(df_genres)):

    for j in range(20):

        if df_genres.iloc[i][j] == 1:

            df_genres.iloc[i][j] = df2['revenue'][i]

            

df_genres.head()
df_genres = df_genres.replace(0, np.NaN)



y = pd.DataFrame(len(df_genres) - df_genres.isnull().sum())

x = pd.DataFrame(df_genres.sum())



z = x / y



type(z.iloc[1,0])



z
import matplotlib.pyplot as plt



z.plot(kind='bar')