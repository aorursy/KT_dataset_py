#!pip install ast

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ast import literal_eval 
#reading the main file

movies=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv',low_memory=False)



# taking a look at on line in the data set

movies.iloc[4]

#size of the data set

dim = movies.shape

print(f"The main movie dataset has {dim[0]} rows and {dim[1]} columns ")

#Let's list all columns name of our data sets

columns =movies.columns.values

print(f"Here is the list of all columns: \n {columns} ")
movies=movies.drop(['poster_path','production_companies','spoken_languages','belongs_to_collection','homepage','production_countries','status','original_title'], axis=1)
movies.dtypes
original_memory=movies.memory_usage(deep=True)

movies.memory_usage(deep=True)
print(movies['adult'].value_counts())
def to_boolean(x):

    '''Take an object value and convert into boolean

       return NaN is the the value is incorrect

    '''

    try:

        x = bool(x)

    except:

        x= np.nan

    return x



#creating a function to convert to int

def to_int(num):

    try:

        num=int(num)

    except:

        num=np.nan

    return num



# converting a column to float

def to_float(num):

    '''Take an object type and convert to float

       return NaN for non numeric values

    '''

    try:

        num=float(num)

    except:

        num=np.nan

    return num



#converting a column as category

def to_category(num):

    '''Take an object type and convert to categorical

       return NaN if convertion fails

    '''

    try:

        num=category(num)

    except:

        num=np.nan

    return num



#converting a column as int

def to_int(num):

    '''Take an object type and convert to int

       return NaN if convertion fails

    '''

    try:

        num=int(num)

    except:

        num=np.nan

    return num
#converting the adult column in movie boolean

movies['adult']=movies['adult'].apply(to_boolean)



#converting the video column in movie boolean

movies['video']=movies['video'].apply(to_boolean)





#converting the budget column to float

movies['budget']=movies['budget'].apply(to_float)



#converting the popularity column to float

movies['popularity']=movies['popularity'].apply(to_float)



#converting the budget column to float

movies['revenue']=movies['revenue'].apply(to_float)



#converting the vote_count column to float

movies['vote_count']=movies['vote_count'].apply(to_float)



#converting the vote_average column to float

movies['vote_average']=movies['vote_average'].apply(to_float)



#converting the Id column to int

movies['id']=movies['id'].apply(to_int)



#converting the Id column to categorical

movies['original_language']=movies['original_language'].apply(to_category)



#convert release_date to datetime

movies['release_date']=pd.to_datetime(movies['release_date'], errors='coerce')



print(movies['runtime'].describe())

movies['runtime']=movies['runtime'].apply(to_int)



#getting the current memory size and compare with the original

current_memory=movies.memory_usage(deep=True)

(1-current_memory/original_memory)*100

#movies.dtypes
print(movies.iloc[0]['genres'])

print(type(movies.iloc[0]['genres']))
#let us fill all the genres with non value by an empty list.



movies['genres']=movies['genres'].fillna('[]')



#evaluating the column and return the right object

movies['genres']=movies['genres'].apply(literal_eval)



#then convert the colum to a list of genres

movies['genres']=movies['genres'].apply(lambda genre : [x['name'] for x in genre] if isinstance(genre,list) else [])

movies.iloc[0]['genres']
tempgenre=movies.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1,drop=True)

tempgenre.name='genre'

tempgenre.head()
# new data set

movie_gen=movies.drop('genres', axis=1).join(tempgenre)

movie_gen.head(2)
print(f' The new dataset has {movie_gen.shape[0]} rows and {movie_gen.shape[1]} columns')
credits = pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')

credits.head(3)
credits['crew'].fillna('[]', inplace=True)

credits['cast'].fillna('[]', inplace=True)

credits['crew']=credits['crew'].apply(literal_eval)

credits['cast']=credits['cast'].apply(literal_eval)

def get_actor(namedict):

    result={}

    try:

        result['actor_1_name']=[x['name'] for x in namedict if x['order']==0][0]

        result['actor_2_name']=[x['name'] for x in namedict if x['order']==1][0]

        result['actor_3_name']=[x['name'] for x in namedict if x['order']==2][0]

               

    except:

        name=np.nan

    return result

        

def get_director(namedict):

    try:

        name=[x['name'] for x in namedict if x['job']=='Director'][0]

    except:

        name=np.nan

    return name

        

#Creating two additional  columns for director and actor    

credits['director']=credits['crew'].apply(get_director)



#creating the 3 main actors for the movie

credits['actor_1_name']   =credits['cast'].apply(get_actor).apply(lambda x :x.get('actor_1_name',''))

credits['actor_2_name']   =credits['cast'].apply(get_actor).apply(lambda x :x.get('actor_2_name',''))

credits['actor_3_name']   =credits['cast'].apply(get_actor).apply(lambda x :x.get('actor_3_name',''))

#Dropping cast and cast and crew columns as no longer needed

credits.drop(['cast','crew'], inplace=True, axis=1)
#Merging the two new columns to the main dataset

combine_data =pd.merge(movie_gen,credits, on='id', how='inner')

combine_data.head(3)
combine_data.to_csv('combine_data.csv')