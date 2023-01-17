import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
movies_frame = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', header = 0, low_memory = False)
pd.set_option('display.max_columns', None)

movies_frame.head()
columns = ['budget', 'genres', 'id', 'popularity', 'production_companies', 'release_date', 'revenue', 'runtime',

          'title']
movies_frame = movies_frame[columns]
movies_frame.head()
movies_frame.info()
#Handle missing values in all columns |

## Handle data types (this is new task) |

#Round popularity with 2 points |

#Extract production name from col |

#Rename 'id' col to 'movieId'

#Keep only ralease year or better add new col with only realease year |

#Divide reveune to mln |

#Divide budget to mln |

#Check all column values |

#Extract generes
#for the sake of convenience

df = movies_frame.copy()
del movies_frame
#Handle missing values in all columns
df.info()
df.dropna(inplace = True)
df.info()
#Budget should be int of float
df.loc[df['budget'].str.contains(r'\D') == True]

#checking if there is any row containing non digit
df['budget'] = pd.to_numeric(df['budget'])
df['budget'].dtype
#'Id' should be numeric int

df.loc[df['id'].str.contains(r'\D') == True]
df['id'] = pd.to_numeric(df['id'])
df['id'].dtype
#'popularity' too

df.loc[df['popularity'].str.contains(r'\d') == False]
df['popularity'] = pd.to_numeric(df['popularity'])
df.popularity.dtype
#And I'll round it

df['popularity'] = df['popularity'].round()
df['popularity'].iloc[:5]
#I'll check 'revenue' and 'runtime' for some strange values
df.loc[df['revenue'].apply(lambda x: True if r'D' in str(x) else False) == True]
df.loc[df['runtime'].apply(lambda x: True if r'D' in str(x) else False) == True]
df.info()
#I have finished with numeric features
#As 'title' is the only col I'll not work with later, I'll change only it`s type

#Other object cols I'll manage later
df['title'] = df['title'].apply(str)
df.loc[df['title'].apply(lambda x: True if x.isdigit() else False) == True]
#As we can see there is some titles which contains only digits, that is a little bit strange, but it can be. So I`ll keep them

#for further EDA

#I noticed another interesting detail, there are some films with 0 budget and 0 revenue. I'll keep this in my mind during EDA
#Extract production name from col
#Definitly there are a lot of companies in the world, so I'll make from this col something like categorical feature

#with some most popular studios
df['production_companies'].value_counts(ascending = False).iloc[:20]
#Most popular studios

studios = ['Metro-Goldwyn-Mayer (MGM)', 'Warner Bros.', 'Paramount Pictures', 'Twentieth Century Fox Film Corporation',

          'Universal Pictures']
for studio in studios:

    df['production_companies'] = df['production_companies'].apply(lambda x: studio

                                                              if studio in x else x)
df['production_companies'] = df['production_companies'].apply(lambda x: 'other' if x not in studios else x) 
df['production_companies'].value_counts()
#Rename 'id' col to 'movieId'
df.rename(columns = {'id': 'movieId'}, inplace = True)
#Keep only release year or better add new col with only realease year
df['release_year'] = df['release_date'].apply(lambda x: str(x)[:4])
df['release_year'].iloc[:5]
#Change release date to datetime
df['release_date'] = pd.to_datetime(df['release_date'], format = '%Y-%m-%d')
df.dtypes
#Divide revenue to mln

#Divide budget to mln
df['revenue'] = df['revenue'].div(1000000)
df['budget'] = df['budget'].div(1000000)
#Check all column values
#I think this is more like an EDA task so I'll leave this for later
df.info()
#Extract generes
df.head()
test_df = df.copy()
test_df['genres'] = test_df['genres'].str.replace("'id'", '')

test_df['genres'] = test_df['genres'].str.replace("'name'", '')

test_df['genres'] = test_df['genres'].str.replace(":", '')

test_df['genres'] = test_df['genres'].str.replace("\d", '')

test_df['genres'] = test_df['genres'].str.replace(pat = ',', repl = '')

test_df['genres'] = test_df['genres'].str.replace(pat = '{', repl = '')

test_df['genres'] = test_df['genres'].str.replace(pat = '}', repl = '')
genres_df = test_df['genres'].str.split(pat = "'", expand = True)

genres_df.head()
genres = genres_df[1].value_counts().index.copy()
#My final goal was to find out most popular genres
del test_df

del genres_df
for genre in genres:

    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)
#df.head()
df.drop(columns = ['genres'], inplace = True)
df.head()
df.info()
df['release_year'] = pd.to_numeric(df['release_year'])

#Will be more easy to work with int rather than with datetime

#*Surely I understand that datetime will work faster, but in this dataset it is not significant
df['release_year'].dtype
#There are two things I forgot

#1) Check duplicates

#2) Round 'revenue' and 'budget'
#2) Round 'revenue' and 'budget'

df['revenue'] = df['revenue'].round(2)
df['budget'] = df['budget'].round(2)
#1) Check duplicates
df.drop_duplicates(keep = 'first', inplace = True)
#df.to_csv(r'C:\Users\Gev\Desktop\Data science\Movies\EDA\EDA_data.csv')