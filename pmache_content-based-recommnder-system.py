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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
movies_df = pd.read_csv('/kaggle/input/movies/movies.csv', sep='::', names=['MovieID','Title','Genres'])

ratings_df = pd.read_csv('/kaggle/input/ratings/ratings.csv', sep='::', names=['UserID','MovieID','Rating','Timestamp'])

users_df = pd.read_csv('/kaggle/input/users/users.csv', sep='::',names=['UserID','Gender','Age','Occupation','Zip-code'])
print('Shape of movies dataset {}'.format(movies_df.shape))

print('Shape of ratings dataset {}'.format(ratings_df.shape))

print('Shape of users dataset {}'.format(users_df.shape))
movies_df.head()
ratings_df.head()
users_df.head()
movies_df['Year'] = movies_df['Title'].str.extract('(\(\d\d\d\d\))')

movies_df['Year'] = movies_df['Year'].str.extract('(\d\d\d\d)')

movies_df['Title'] = movies_df['Title'].str.replace('(\(\d\d\d\d\))','')
movies_df['Title'] = movies_df['Title'].apply(lambda title : title.strip())
movies_df['Genres'] = movies_df['Genres'].apply(lambda genres : genres.split('|'))
movies_df.head()
moviesWithGenres_df = movies_df.copy()
for index, row in movies_df.iterrows():

    for genre in row['Genres']:

        moviesWithGenres_df.at[index, genre] = 1

        
moviesWithGenres_df.head()
moviesWithGenres_df.fillna(0, inplace=True)
ratings_df.head()
ratings_df.drop('Timestamp', axis=1, inplace=True)
user_12_ratings = ratings_df[ratings_df['UserID'] == 12]

user_12_ratings.head()
user_12_ratings = pd.merge(user_12_ratings, moviesWithGenres_df, on='MovieID')

user_12_ratings.head()
user_12_genre = user_12_ratings.drop(columns=['MovieID', 'Rating','UserID','Title','Genres','Year'], axis=1)

user_12_genre
user_12_profile = user_12_genre.transpose().dot(user_12_ratings['Rating'])
plt.figure(figsize=(10,8))

sns.barplot(data= user_12_profile.reset_index().sort_values(by=0, ascending =False), x = 'index', y=0)

plt.title('Genres preferred by user 12', fontsize=24)

plt.xticks(rotation=90, fontsize=12)

plt.ylabel('Percentage',fontsize=16)

plt.xlabel('Genre',fontsize=16)
genre_table = moviesWithGenres_df.drop(columns=['Title', 'Genres','Year','MovieID'],axis=1)

genre_table.head()
recommendation_user_12 = genre_table * user_12_profile

recommendation_user_12.head()
recommendation_user_12 = recommendation_user_12.sum(axis=1)/user_12_profile.sum()
recommendation_user_12 = pd.DataFrame(recommendation_user_12)

recommendation_user_12 = recommendation_user_12.reset_index()

recommendation_user_12.rename(columns = {'index':'MovieID', 0:'Recommend_Percent'},inplace=True)

recommendation_user_12 = recommendation_user_12.sort_values(by='Recommend_Percent',ascending=False)

recommendation_user_12.head(10)
recommendation_user_12 = pd.merge(recommendation_user_12,movies_df, on='MovieID')

recommendation_user_12.head(10)
def get_user_profile(userID):

    '''

       Input required: Id of the user

       Returns user profile in the form of pandas Series object. 

       User profile is percentage of each genre rated/liked by user

       

    '''

    userID_ratings = ratings_df[ratings_df['UserID'] == userID]

    userID_ratings = pd.merge(userID_ratings, moviesWithGenres_df, on='MovieID')

    userID_genre = userID_ratings.drop(columns=['MovieID', 'Rating','UserID','Title','Genres','Year'], axis=1)

    user_profile = userID_genre.transpose().dot(userID_ratings['Rating'])

    

    return user_profile
# test above function

get_user_profile(12)
def get_recommendation_for_user(user_ID, number_of_movies=10):

    '''

        Returns movies with recommendation percentage in the form of pandas dataframe

        

    '''

    user_profile=  get_user_profile(user_ID)

    recommendation_for_user = genre_table * user_profile

    recommendation_for_user = recommendation_for_user.sum(axis=1)/user_12_profile.sum()

    recommendation_for_user = pd.DataFrame(recommendation_for_user, columns=['Recommend_Percent'])

    recommendation_for_user.index.name='idx'

    recommendation_for_user.reset_index(inplace=True)

    recommendation_for_user.rename(columns={'idx':"MovieID"}, inplace=True)

    recommendation_for_user = recommendation_for_user.sort_values(by='Recommend_Percent',ascending=False)

    recommendation_for_user = recommendation_for_user.head(number_of_movies)

    

    recommendation_for_user = pd.merge(recommendation_for_user,movies_df, on='MovieID')

    return recommendation_for_user
# test above function for some users

get_recommendation_for_user(12,5)
get_recommendation_for_user(25)
get_recommendation_for_user(311)