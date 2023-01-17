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
#import the libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import Series



%matplotlib inline

pd.set_option('display.max_rows', 500)
movies = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv', index_col='ID')

#found a way to remove the annoying 'Unnamed: 0'

movies = movies.drop('Unnamed: 0', axis =1)

movies.head()
#function to investigate the data



def data_inv(df):

    print('No of Rows: ', df.shape[0])

    print('No of Coloums: ', df.shape[1])

    print('**'*25)

    print('Colums Names: \n', df.columns)

    print('**'*25)

    print('Datatype of Columns: \n', df.dtypes)

    print('**'*25)

    print('Missing Values: ')

    c = df.isnull().sum()

    c = c[c>0]

    print(c)

    print('**'*25)

    print('Missing vaules %age wise:\n')

    print((100*(df.isnull().sum()/len(df.index))))

    print('**'*25)

    print('Pictorial Representation:')

    plt.figure(figsize=(8,6))

    sns.heatmap(df.isnull(), yticklabels=False,cbar=False)

    plt.show()  

data_inv(movies)
movies.drop(['Rotten Tomatoes', 'Age'], axis = 1, inplace=True)
data_inv(movies)
movies.dropna(subset=['IMDb'], inplace=True)
data_inv(movies)
movies.dropna(subset=['Directors', 'Genres', 'Country', 'Language', 'Runtime'],inplace=True)

data_inv(movies)
movies.head()
movies.shape
#checking for any duplicate data

movies.drop_duplicates(inplace=True)
#lets check out the unique values in first Column

movies['Title'].value_counts()
#checking years

plt.figure(figsize=(10,10))

sns.distplot(movies['Year'])

plt.show()
#Writing a function to calculate the movies in different platforms

def movies_count(platform, count=False):

    if count==False:

        print('Movies in {} are {}'. format(platform, movies[platform].sum()))

    else:

        return movies[platform].sum()

    
movies['IMDb']
plt.figure(figsize=(10,10))

sns.distplot(movies['IMDb'])

#plt.hist(movies['IMDb'])

plt.show()
print('Minimum IMDb rating: ', movies['IMDb'].min())

print('Maximum IMDb rating: ', movies['IMDb'].max())
movies_count('Netflix')

movies_count('Hulu')

movies_count('Prime Video')

movies_count('Disney+')
#Movies per platform

labels = 'Netflix', 'Hulu', 'Prime Video', 'Disney'

size = [movies_count('Netflix', count=True), 

        movies_count('Hulu', count=True),

        movies_count('Prime Video', count=True),

        movies_count('Disney+', count=True)]



explode = (0.1, 0.1, 0.1, 0.1)



#plotting

fig1, ax1 = plt.subplots()



ax1.pie(size,

       labels = labels,

       autopct = '%1.1f%%',

       explode = explode,

       shadow = True,

       startangle = 100)



ax1.axis = ('equal')

plt.show()
movies['Type'].value_counts()
movies.drop('Type', inplace=True, axis =1)

movies.shape
movies['Directors'].value_counts()
s = movies['Directors'].str.split(',').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'Directors'

del movies['Directors']

df_directors = movies.join(s)
data = df_directors['Directors'].value_counts()

threshold = 10

prob = data > threshold

data_new = data.loc[prob]



plt.figure(figsize=(10,10))

data_new.plot(kind='bar')

plt.show()
s = movies['Genres'].str.split(',').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'Genres'

del movies['Genres']

df_genres = movies.join(s)
df_genres.head()
plt.figure(figsize=(10,10))

sns.countplot(x='Genres', data=df_genres)

plt.xticks(rotation=90)

plt.show()
s = movies['Country'].str.split(',').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'Country'

del movies['Country']

df_country = movies.join(s)
df_country['Country'].value_counts()[:10].plot(kind='bar')

plt.show()
s = movies['Language'].str.split(',').apply(Series,1).stack()

s.index = s.index.droplevel(-1)

s.name = 'Language'

del movies['Language']

df_language = movies.join(s)
df_language['Language'].value_counts()[:10].plot(kind='barh')

plt.show()
sns.distplot(movies['Runtime'])

plt.show()
filter =( movies['Runtime'] == (movies['Runtime'].max()))

movies[filter]
filter =( movies['Runtime'] == (movies['Runtime'].min()))

movies[filter]
#top rated movie

filter =( movies['IMDb'] == (movies['IMDb'].max()))

movies[filter]
# Bottom Rated movie

filter =( movies['IMDb'] == (movies['IMDb'].min()))

movies[filter]