#load the data and merge test and train set 

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_submission = pd.read_csv ('../input/sample_submission.csv')



#merge datasets for faster feature engineering 
df_tmdb = df_train.append(df_test,ignore_index=True)
df_tmdb.shape
df_tmdb.head(5)
df_tmdb.tail(5)
#insert some libraries
import json
import ast
from pprint import pprint
import seaborn as sns 
from scipy.stats import norm,skew
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
pd.set_option('display.max_columns', None)


def process_homepage(homepage):
    """
    Returns 0 if homepage is empty and 1 otherwise
    """
    if pd.isnull(homepage):
        return 0
    return 1

def process_cast(cast):
    """
    Extracts cast members names and returns string
    containing these names delimited by ;
    """
    if pd.isnull(cast):
        return ''
    cast_members = ast.literal_eval(cast)
    members_names = [member['name'] for member in cast_members]
    return ';'.join(members_names)
def extract_genres(genres, genres_list):
    """
    Extracts genres from genres json
    """
    if not pd.isnull(genres):
        parsed_genres = ast.literal_eval(genres)
        genres_list.update([genre['name'] for genre in parsed_genres])

def process_row(row, genres_list, df):
    """
    Processes one row of the dataframe. Extracts genres,
    fill columns with values for genres of the row, fill
    homepage with 0 or 1 by the value in default homepage,
    calculates average movie star power for the row(movie).
    """
    if pd.isnull(row['genres']):
        parsed_genres = []
    else:
        parsed_genres = ast.literal_eval(row['genres']) # parsed_genres from the json
    homepage = process_homepage(row['homepage']) # 0 or 1 for the homepage
    if len(row['cast_names']) > 0:
        actors_list = row['cast_names'].split(';') # list of actors of the movie
        total_star_power = 0
        total_star_power_before = 0
        for actor in actors_list:
            appearances = df[df['cast_names'].str.contains(actor)] # only if actor appears in the movie
            before = appearances[appearances['release_date'] <= row['release_date']] # only if actor apears BEFORE or ON release date of the movie
            total_star_power += len(appearances)
            total_star_power_before += len(before)
        avg_star_power = total_star_power_before / len(actors_list) # calculate average star power BEFORE and ON release
        avg_star_power_total = total_star_power / len(actors_list) # calculate average star power without taking release date into account
    else:
        avg_star_power = 0
        avg_star_power_total = 0
    new_row = {'id': row['id'], 'homepage': homepage,
            'avg_star_power': avg_star_power,
            'avg_star_power_total': avg_star_power_total}
    for genre in parsed_genres:
        new_row[genre['name']] = 1
    return pd.Series(data=new_row)

def get_all_genres(df):
    """
    Returns all genres of the films in dataframe
    """
    genres_list = set()
    df['genres'].apply(extract_genres, args=(genres_list,))
    return genres_list


def parse_database(df):
    genres_list = list(get_all_genres(df)) # gets all genres, needed for the column names
    columns = ['id', 'cast_names', 'avg_star_power', 'avg_star_power_total','homepage']
    columns.extend(genres_list)
    parsed_df = pd.DataFrame(columns=columns) # new dataframe that will be outputed

    df['cast_names'] = df['cast'].apply(process_cast) # parse cast members

    parsed_df = df.apply(process_row, axis=1, args=(genres_list, df)) # process rows of the dataframe

    parsed_df = parsed_df.fillna(value=0) # fill NaN with zeros
    columns.remove('cast_names') # remove parsed cast names
    parsed_df = parsed_df[columns] # order columns

    return parsed_df


df_tmdb_adjusted = parse_database(df_tmdb)
df_tmdb_adjusted.head(5)
#drop original column 'homepage'
for columns in ['homepage']:
    df_tmdb.drop(columns, axis=1, inplace=True)
#merge columns homepage, different genres, avg_star_power and _avg_star_power_total with df_tmdb
df_tmdb = pd.merge(df_tmdb, df_tmdb_adjusted, on='id')
df_tmdb.head(5)
def process_belongs_to_collection(belongs_to_collection):
    """
    Returns 0 if belongs_to_collection is empty and 1 otherwise
    """
    if pd.isnull(belongs_to_collection):
        return 0
    return 1

def process_crew(crew):
    """
    Extracts crew members names and returns string
    containing these names delimited by ;
    """
    if pd.isnull(crew):
        return ''
    crew_members = ast.literal_eval(crew)
    members_names = [member['name'] for member in crew_members]
    return ';'.join(members_names)


def extract_spoken_languages(spoken_languages, spoken_languages_list):
    """
    Extracts spoken languages from json
    """
    if not pd.isnull(spoken_languages):
        parsed_spoken_languages = ast.literal_eval(spoken_languages)
        spoken_languages_list.update([spoken_languages['name'] for spoken_languages in parsed_spoken_languages])

def process_row(row, spoken_languages_list, df2):
    """
    Processes one row of the dataframe. Extracts spoken_languages,
    fill columns with values for spoken_languages of the row, fill
    belongs_to_collection with 0 or 1,
    calculates average crew star power for the row(movie).
    """
    if pd.isnull(row['spoken_languages']):
        parsed_spoken_languages = []
    else:
        parsed_spoken_languages = ast.literal_eval(row['spoken_languages']) # parsed_spoken_languages from the json
    belongs_to_collection = process_belongs_to_collection(row['belongs_to_collection']) # 0 or 1 for belongs_to_collection
    if len(row['crew_names']) > 0:
        crews_list = row['crew_names'].split(';') # list of crew of the movie
        total_crew_power = 0
        total_crew_power_before = 0
        for crew in crews_list:
            appearances = df2[df2['crew_names'].str.contains(crew)] # only if crew member appears in the movies
            before = appearances[appearances['release_date'] <= row['release_date']] # and only on or before release date of the movie
            total_crew_power += len(appearances)
            total_crew_power_before += len(before)
        avg_crew_power = total_crew_power_before / len(crews_list) # calculate average star power BEFORE and ON release
        avg_crew_power_total = total_crew_power / len(crews_list) # calculate average star power without counting in release date
    else:
        avg_crew_power = 0
        avg_crew_power_total = 0
    new_row = {'id': row['id'], 'belongs_to_collection': belongs_to_collection,
            'avg_crew_power': avg_crew_power,
            'avg_crew_power_total': avg_crew_power_total}
    for spoken_languages in parsed_spoken_languages:
        new_row[spoken_languages['name']] = 1
    return pd.Series(data=new_row)
def get_all_spoken_languages(df2):
    """
    Returns all spoken_languages of the films in dataframe
    """
    spoken_languages_list = set()
    df2['spoken_languages'].apply(extract_spoken_languages, args=(spoken_languages_list,))
    return spoken_languages_list
def parse_database(df2):
    spoken_languages_list = list(get_all_spoken_languages(df2)) # gets all spoken_languages, needed for the column names
    columns = ['id', 'crew_names', 'avg_crew_power', 'avg_crew_power_total','belongs_to_collection']
    columns.extend(spoken_languages_list)
    parsed_df2 = pd.DataFrame(columns=columns) # new dataframe that will be outputed

    df2['crew_names'] = df2['crew'].apply(process_cast) # parse crew members

    parsed_df2 = df2.apply(process_row, axis=1, args=(spoken_languages_list, df2)) # process rows of the dataframe

    parsed_df2 = parsed_df2.fillna(value=0) # fill NaN with zeros
    columns.remove('crew_names') # remove parsed cast names
    parsed_df2 = parsed_df2[columns] # order columns

    return parsed_df2
df_tmdb_adjusted2 = parse_database(df_tmdb)
df_tmdb_adjusted2.head(5)
#drop original column 'belongs_to_collection'
for columns in ['belongs_to_collection']:
    df_tmdb.drop(columns, axis=1, inplace=True)
   
#RECODE SPOKENLANGUAGES TO ONLY ENGLISH, MIXED or OTHER. FIRST RECODE ENGLISH TO 100
def english(series):
    if series == 1:
        return 100
    else:
        return series
    

df_tmdb_adjusted2['English'] = df_tmdb_adjusted2['English'].apply(english)
df_tmdb_adjusted2.head(5)
#then sum over all spoken languages
df_tmdb_adjusted2['languages']= df_tmdb_adjusted2[list(df_tmdb_adjusted2.columns[4:68])].sum(axis=1)
df_tmdb_adjusted2.head(5)
#then recode into three categories 
def spoken_languages(series):
    if series == 100:
        return 'en'
    if series > 100:
        return 'en and other'
    else:
        return 'other'
    

df_tmdb_adjusted2['spoken_langauages'] = df_tmdb_adjusted2['languages'].apply(spoken_languages)
df_tmdb_adjusted2.head(5)
df_tmdb_adjusted2= df_tmdb_adjusted2[['id', 'avg_crew_power', 'avg_crew_power_total', 'belongs_to_collection','spoken_langauages']]

df_tmdb_adjusted2.head(2)
#merge columns belongs_to_collection, different languagues, avg_crew_power and _avg_crew_power_total with df_tmdb
df_tmdb = pd.merge(df_tmdb, df_tmdb_adjusted2, on='id')
#extract Keywords from Json column
df_tmdb['Keywords'] = df_tmdb['Keywords'].apply(lambda x: [''] if pd.isna(x) else [str(j['name']) for j in (eval(x))])
def process_production(production_companies):
    """
    Extracts production_companies and returns string
    containing these names delimited by ;
    """
    if pd.isnull(production_companies):
        return ''
    production_members = ast.literal_eval(production_companies)
    production_names = [member['name'] for member in production_members]
    return ';'.join(production_names)

def process_row(row, df3):
    """
    Processes one row of the dataframe. Extracts production_companies
   calculates average production star power for the row(movie).
    """
    if len(row['production_names']) > 0:
        production_list = row['production_names'].split(';') # list of production_companies of the movie
        total_production_power = 0
        total_production_power_before = 0
        for production_companies in production_list:
            appearances = df3[df3['production_names'].str.contains(production_companies)] # only if production company appears in the movie
            before = appearances[appearances['release_date'] <= row['release_date']] # and only on or before release date of the movie
            total_production_power += len(appearances)
            total_production_power_before += len(before)
        avg_production_power = total_production_power_before / len(production_list) # calculate average production power BEFORE and ON release
        avg_production_power_total = total_production_power / len(production_list) # calculate average production power without counting in release date
    else:
        avg_production_power = 0
        avg_production_power_total = 0
    new_row = {'id': row['id'],
            'avg_production_power': avg_production_power,
            'avg_production_power_total': avg_production_power_total}
    return pd.Series(data=new_row)
def parse_database(df3):
    columns = ['id', 'production_names', 'avg_production_power', 'avg_production_power_total']
    parsed_df3 = pd.DataFrame(columns=columns) # new dataframe that will be outputed

    df3['production_names'] = df3['production_companies'].apply(process_production) # parse production companies

    parsed_df3 = df3.apply(process_row, axis=1, args=(df3,)) # process rows of the dataframe

    parsed_df3 = parsed_df3.fillna(value=0) # fill NaN with zeros
    columns.remove('production_names') # remove parsed production names
    parsed_df3 = parsed_df3[columns] # order columns

    return parsed_df3
df_tmdb_adjusted3 = parse_database(df_tmdb)
df_tmdb_adjusted3.head()
#merge columns avg_production_power and avg_production_power_total with df_tmdb
df_tmdb = pd.merge(df_tmdb, df_tmdb_adjusted3, on='id')
df_tmdb.head(5)
#extract production_countries from Json column
df_tmdb['production_countries'] = df_tmdb['production_countries'].apply(lambda x: [''] if pd.isna(x) else [str(j['name']) for j in (eval(x))])

#create column 'production_USA'. 1 is yes, 0 is no
df_tmdb['production_USA'] = df_tmdb['production_countries'].apply(lambda x: 1 if 'United States of America' in x else 0)

df_tmdb.columns
#calculate number of cast, crew, production companies, production countries, keywords and number of letters/words in original title, title, overview and tagline

df_tmdb["cast_len"] = df_tmdb.loc[df_tmdb["cast"].notnull(),"cast"].apply(lambda x : len(x))
df_tmdb["crew_len"] = df_tmdb.loc[df_tmdb["crew"].notnull(),"crew"].apply(lambda x : len(x))
df_tmdb["production_companies_len"]=df_tmdb.loc[df_tmdb["production_companies"].notnull(),"production_companies"].apply(lambda x : len(x))
df_tmdb["production_countries_len"]=df_tmdb.loc[df_tmdb["production_countries"].notnull(),"production_countries"].apply(lambda x : len(x))
df_tmdb["keywords_len"]=df_tmdb.loc[df_tmdb["Keywords"].notnull(),"Keywords"].apply(lambda x : len(x))
df_tmdb["genres_len"]=df_tmdb.loc[df_tmdb["genres"].notnull(),"genres"].apply(lambda x : len(x))
df_tmdb['original_title_letter_count'] = df_tmdb['original_title'].str.len() 
df_tmdb['original_title_word_count'] = df_tmdb['original_title'].str.split().str.len() 
df_tmdb['title_word_count'] = df_tmdb['title'].str.split().str.len()
df_tmdb['overview_word_count'] = df_tmdb['overview'].str.split().str.len()
df_tmdb['tagline_word_count'] = df_tmdb['tagline'].str.split().str.len()

#look at missing values 
df_tmdb.info()
#replace missing values of new columns with 'len' with mode
for column in ['runtime', 'cast_len', 'crew_len', 'production_companies_len', 'genres_len', 'overview_word_count', 'tagline_word_count']:
    df_tmdb[column].fillna(df_tmdb[column].mode()[0], inplace=True)
    
df_tmdb.info()
#describe numeric variables  
df_tmdb.describe()
df_tmdb[df_tmdb.dtypes[(df_tmdb.dtypes=="float64")|(df_tmdb.dtypes=="int64")]
                        .index.values].hist(figsize=[20,20])
#all columns lookd skewed. What are the values?

df_tmdb.skew(axis = 0, skipna = True, numeric_only= True) 
#for some skewed columns, I will add a log transform as a feature. First Popularity (skewness 19.96)
#log transform popularity ( after missing values are)
df_tmdb['log_popularity'] = np.log1p(df_tmdb.popularity)
#skewness down from 14.37 to -2.61
df_tmdb["log_popularity"].skew(axis = 0) 

plt.hist(df_tmdb.log_popularity, bins=20)

#budget. First replace some missing values, found in different kernels.  

df_tmdb.loc[df_tmdb['id'] == 90,'budget'] = 30000000         # Sommersby          
df_tmdb.loc[df_tmdb['id'] == 118,'budget'] = 60000000        # Wild Hogs
df_tmdb.loc[df_tmdb['id'] == 149,'budget'] = 18000000        # Beethoven
df_tmdb.loc[df_tmdb['id'] == 464,'budget'] = 20000000        # Parenthood
df_tmdb.loc[df_tmdb['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
df_tmdb.loc[df_tmdb['id'] == 513,'budget'] = 930000          # From Prada to Nada
df_tmdb.loc[df_tmdb['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
df_tmdb.loc[df_tmdb['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
df_tmdb.loc[df_tmdb['id'] == 850,'budget'] = 90000000        # Modern Times
df_tmdb.loc[df_tmdb['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
df_tmdb.loc[df_tmdb['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   
df_tmdb.loc[df_tmdb['id'] == 1359,'budget'] = 10000000       # Stir Crazy 
df_tmdb.loc[df_tmdb['id'] == 1542,'budget'] = 1500000        # All at Once
df_tmdb.loc[df_tmdb['id'] == 1542,'budget'] = 15800000       # Crocodile Dundee II
df_tmdb.loc[df_tmdb['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
df_tmdb.loc[df_tmdb['id'] == 1714,'budget'] = 46000000       # The Recruit
df_tmdb.loc[df_tmdb['id'] == 1721,'budget'] = 17500000       # Cocoon
df_tmdb.loc[df_tmdb['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
df_tmdb.loc[df_tmdb['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
df_tmdb.loc[df_tmdb['id'] == 2612,'budget'] = 15000000       # Field of Dreams
df_tmdb.loc[df_tmdb['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
df_tmdb.loc[df_tmdb['id'] == 2801,'budget'] = 10000000       # Fracture
df_tmdb.loc[df_tmdb['id'] == 3033,'budget'] = 250 
df_tmdb.loc[df_tmdb['id'] == 3051,'budget'] = 50
df_tmdb.loc[df_tmdb['id'] == 3084,'budget'] = 337
df_tmdb.loc[df_tmdb['id'] == 3224,'budget'] = 4  
df_tmdb.loc[df_tmdb['id'] == 3594,'budget'] = 25  
df_tmdb.loc[df_tmdb['id'] == 3619,'budget'] = 500  
df_tmdb.loc[df_tmdb['id'] == 3831,'budget'] = 3  
df_tmdb.loc[df_tmdb['id'] == 3935,'budget'] = 500  
df_tmdb.loc[df_tmdb['id'] == 4049,'budget'] = 995946 
df_tmdb.loc[df_tmdb['id'] == 4424,'budget'] = 3  
df_tmdb.loc[df_tmdb['id'] == 4460,'budget'] = 8  
df_tmdb.loc[df_tmdb['id'] == 4555,'budget'] = 1200000 
df_tmdb.loc[df_tmdb['id'] == 4624,'budget'] = 30 
df_tmdb.loc[df_tmdb['id'] == 4645,'budget'] = 500 
df_tmdb.loc[df_tmdb['id'] == 4709,'budget'] = 450 
df_tmdb.loc[df_tmdb['id'] == 4839,'budget'] = 7
df_tmdb.loc[df_tmdb['id'] == 3125,'budget'] = 25 
df_tmdb.loc[df_tmdb['id'] == 3142,'budget'] = 1
df_tmdb.loc[df_tmdb['id'] == 3201,'budget'] = 450
df_tmdb.loc[df_tmdb['id'] == 3222,'budget'] = 6
df_tmdb.loc[df_tmdb['id'] == 3545,'budget'] = 38
df_tmdb.loc[df_tmdb['id'] == 3670,'budget'] = 18
df_tmdb.loc[df_tmdb['id'] == 3792,'budget'] = 19
df_tmdb.loc[df_tmdb['id'] == 3881,'budget'] = 7
df_tmdb.loc[df_tmdb['id'] == 3969,'budget'] = 400
df_tmdb.loc[df_tmdb['id'] == 4196,'budget'] = 6
df_tmdb.loc[df_tmdb['id'] == 4221,'budget'] = 11
df_tmdb.loc[df_tmdb['id'] == 4222,'budget'] = 500
df_tmdb.loc[df_tmdb['id'] == 4285,'budget'] = 11
df_tmdb.loc[df_tmdb['id'] == 4319,'budget'] = 1
df_tmdb.loc[df_tmdb['id'] == 4639,'budget'] = 10
df_tmdb.loc[df_tmdb['id'] == 4719,'budget'] = 45
df_tmdb.loc[df_tmdb['id'] == 4822,'budget'] = 22
df_tmdb.loc[df_tmdb['id'] == 4829,'budget'] = 20
df_tmdb.loc[df_tmdb['id'] == 4969,'budget'] = 20
df_tmdb.loc[df_tmdb['id'] == 5021,'budget'] = 40 
df_tmdb.loc[df_tmdb['id'] == 5035,'budget'] = 1 
df_tmdb.loc[df_tmdb['id'] == 5063,'budget'] = 14 
df_tmdb.loc[df_tmdb['id'] == 5119,'budget'] = 2 
df_tmdb.loc[df_tmdb['id'] == 5214,'budget'] = 30 
df_tmdb.loc[df_tmdb['id'] == 5221,'budget'] = 50 
df_tmdb.loc[df_tmdb['id'] == 4903,'budget'] = 15
df_tmdb.loc[df_tmdb['id'] == 4983,'budget'] = 3
df_tmdb.loc[df_tmdb['id'] == 5102,'budget'] = 28
df_tmdb.loc[df_tmdb['id'] == 5217,'budget'] = 75
df_tmdb.loc[df_tmdb['id'] == 5224,'budget'] = 3 
df_tmdb.loc[df_tmdb['id'] == 5469,'budget'] = 20 
df_tmdb.loc[df_tmdb['id'] == 5840,'budget'] = 1 
df_tmdb.loc[df_tmdb['id'] == 5960,'budget'] = 30
df_tmdb.loc[df_tmdb['id'] == 6506,'budget'] = 11 
df_tmdb.loc[df_tmdb['id'] == 6553,'budget'] = 280
df_tmdb.loc[df_tmdb['id'] == 6561,'budget'] = 7
df_tmdb.loc[df_tmdb['id'] == 6582,'budget'] = 218
df_tmdb.loc[df_tmdb['id'] == 6638,'budget'] = 5
df_tmdb.loc[df_tmdb['id'] == 6749,'budget'] = 8 
df_tmdb.loc[df_tmdb['id'] == 6759,'budget'] = 50 
df_tmdb.loc[df_tmdb['id'] == 6856,'budget'] = 10
df_tmdb.loc[df_tmdb['id'] == 6858,'budget'] =  100
df_tmdb.loc[df_tmdb['id'] == 6876,'budget'] =  250
df_tmdb.loc[df_tmdb['id'] == 6972,'budget'] = 1
df_tmdb.loc[df_tmdb['id'] == 7079,'budget'] = 8000000
df_tmdb.loc[df_tmdb['id'] == 7150,'budget'] = 118
df_tmdb.loc[df_tmdb['id'] == 6506,'budget'] = 118
df_tmdb.loc[df_tmdb['id'] == 7225,'budget'] = 6
df_tmdb.loc[df_tmdb['id'] == 7231,'budget'] = 85
df_tmdb.loc[df_tmdb['id'] == 5222,'budget'] = 5
df_tmdb.loc[df_tmdb['id'] == 5322,'budget'] = 90
df_tmdb.loc[df_tmdb['id'] == 5350,'budget'] = 70
df_tmdb.loc[df_tmdb['id'] == 5378,'budget'] = 10
df_tmdb.loc[df_tmdb['id'] == 5545,'budget'] = 80
df_tmdb.loc[df_tmdb['id'] == 5810,'budget'] = 8
df_tmdb.loc[df_tmdb['id'] == 5926,'budget'] = 300
df_tmdb.loc[df_tmdb['id'] == 5927,'budget'] = 4
df_tmdb.loc[df_tmdb['id'] == 5986,'budget'] = 1
df_tmdb.loc[df_tmdb['id'] == 6053,'budget'] = 20
df_tmdb.loc[df_tmdb['id'] == 6104,'budget'] = 1
df_tmdb.loc[df_tmdb['id'] == 6130,'budget'] = 30
df_tmdb.loc[df_tmdb['id'] == 6301,'budget'] = 150
df_tmdb.loc[df_tmdb['id'] == 6276,'budget'] = 100
df_tmdb.loc[df_tmdb['id'] == 6473,'budget'] = 100
df_tmdb.loc[df_tmdb['id'] == 6842,'budget'] = 30


#replace budgets < 1000 with mean 
mean = df_tmdb.loc[df_tmdb['budget']<1000, 'budget'].mean()
df_tmdb.loc[df_tmdb.budget <1000, 'budget'] = np.nan
df_tmdb.fillna(mean,inplace=True)
#log transform budget. Skewness now between -1 and +1

df_tmdb['log_budget'] = np.log1p(df_tmdb.budget)
df_tmdb["log_budget"].skew(axis = 0) 
plt.hist(df_tmdb.log_budget, bins=20)

#replacing some missing revenue values 
df_tmdb.loc[df_tmdb['id'] == 16,'revenue'] = 171052         # Skinning
df_tmdb.loc[df_tmdb['id'] == 117,'revenue'] = 55287687      # Back to 1942
df_tmdb.loc[df_tmdb['id'] == 270,'revenue'] = 20018         # Glass: A Portrait of Philip in Twelve Parts 
df_tmdb.loc[df_tmdb['id'] == 281,'revenue'] = 10655191      # Bats
df_tmdb.loc[df_tmdb['id'] == 151,'revenue'] = 18000000      # Windwalker
df_tmdb.loc[df_tmdb['id'] == 313,'revenue'] = 11540112      # The Cookout 
df_tmdb.loc[df_tmdb['id'] == 1282,'revenue'] = 46800000     # Death at a Funeral
df_tmdb.loc[df_tmdb['id'] == 451,'revenue'] = 12291275      # Chasing Liberty
df_tmdb.loc[df_tmdb['id'] == 1865,'revenue'] = 181185387    # Scooby-Doo 2: Monsters Unleashed
df_tmdb.loc[df_tmdb['id'] == 2491,'revenue'] = 6849998      # Never Talk to Strangers
df_tmdb.loc[df_tmdb['id'] == 2252,'revenue'] = 51119758     # Bodyguard

mean = df_tmdb.loc[df_tmdb['revenue']<1000, 'revenue'].mean()
df_tmdb.loc[df_tmdb.revenue <1000, 'revenue'] = np.nan
df_tmdb.fillna(mean,inplace=True)
#since regression analysis is based on normal distribution, also log transform revenue
df_tmdb['log_revenue'] = np.log1p(df_tmdb.revenue)
plt.hist(df_tmdb.log_revenue, bins=20)

df_tmdb.head(1)
# also add log of some other features
df_tmdb['log_avg_star_power']= np.log1p(df_tmdb.avg_star_power)
df_tmdb['log_avg_star_power_total'] = np.log1p(df_tmdb.avg_star_power_total)
df_tmdb['log-avg_crew_power'] = np.log1p(df_tmdb.avg_crew_power)
df_tmdb['log_avg_crew_power_total'] = np.log1p(df_tmdb.avg_crew_power_total)
df_tmdb['log_avg_production_power'] = np.log1p(df_tmdb.avg_production_power)
df_tmdb['log_avg_production_power_total'] = np.log1p(df_tmdb.avg_production_power_total)

#log transformation decreased skewness of 'star-, crew, - and production power indices'
df_tmdb.skew(axis = 0, skipna = True, numeric_only= True) 
#Extract month, day and year of the release of the movie 
df_tmdb[['release_month','release_day','release_year']]=df_tmdb['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

df_tmdb.loc[ (df_tmdb['release_year'] <= 19) & (df_tmdb['release_year'] < 100), "release_year"] += 2000
df_tmdb.loc[ (df_tmdb['release_year'] > 19)  & (df_tmdb['release_year'] < 100), "release_year"] += 1900

releaseDate = pd.to_datetime(df_tmdb['release_date']) 
df_tmdb['release_month'] = releaseDate.dt.month
df_tmdb['release_dayofweek'] = releaseDate.dt.dayofweek
df_tmdb['release_quarter'] = releaseDate.dt.quarter


#drop some unneccessary columns 
for columns in ["genres", "imdb_id", "original_language", "original_title", "overview", "poster_path", "production_companies",
               "production_countries", "spoken_languages", "status", "tagline", "title", "Keywords", "cast", "crew",
               "cast_names", "crew_names", "production_names"]:
    df_tmdb.drop(columns, axis=1, inplace=True)
df_tmdb.head(2)
#also drop revenue since log revenue will be predicted. Calculate expm1 before submission
for columns in ["revenue"]:
    df_tmdb.drop(columns, axis=1, inplace=True)
#make dummy release_month
dummies = pd.get_dummies(df_tmdb['release_month']).rename(columns=lambda x: 'release_month' + str(x))
df_tmdb = pd.concat([df_tmdb, dummies], axis=1)

#dummy release_day
dday = pd.get_dummies(df_tmdb['release_dayofweek']).rename(columns=lambda x: 'release_dayofweek' + str(x))
df_tmdb = pd.concat([df_tmdb, dday], axis=1)


#dummy spoken_languages
languagedum= pd.get_dummies(df_tmdb['spoken_langauages']).rename(columns=lambda x: 'languages' + str(x))
df_tmdb = pd.concat([df_tmdb,languagedum], axis=1)

#dummy release_quarter
rq= pd.get_dummies(df_tmdb['release_quarter']).rename(columns=lambda x: 'release_quarter' + str(x))
df_tmdb = pd.concat([df_tmdb,rq], axis=1)
df_tmdb.rename(columns={'log-avg_crew_power':'log_avg_crew_power'},inplace=True) 
#standardize the features
from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(df_tmdb[['budget', 'popularity', 'runtime', 'avg_star_power', "avg_star_power_total", 
                                             "avg_crew_power", "avg_crew_power_total", "avg_production_power", "avg_production_power_total",
                                             "cast_len", "crew_len", "production_companies_len", "production_countries_len", "keywords_len",
                                             "genres_len", "original_title_letter_count", "original_title_word_count", "title_word_count",
                                             "overview_word_count", "tagline_word_count", "log_popularity", "log_budget", "release_year", 
                                             "log_avg_star_power", "log_avg_star_power_total", "log_avg_crew_power", "log_avg_crew_power_total", 
                                             "log_avg_production_power", "log_avg_production_power_total"]])
df_tmdb[['budget', 'popularity', 'runtime', 'avg_star_power', "avg_star_power_total", 
        "avg_crew_power", "avg_crew_power_total", "avg_production_power", "avg_production_power_total",
        "cast_len", "crew_len", "production_companies_len", "production_countries_len", "keywords_len",
        "genres_len", "original_title_letter_count", "original_title_word_count", "title_word_count",
        "overview_word_count", "tagline_word_count", "log_popularity", "log_budget", "release_year",
        "log_avg_star_power", "log_avg_star_power_total", "log_avg_crew_power", "log_avg_crew_power_total", 
        "log_avg_production_power", "log_avg_production_power_total"]] = StandardScaler().fit_transform(df_tmdb[['budget',
        'popularity', 'runtime', 'avg_star_power', "avg_star_power_total", 
        "avg_crew_power", "avg_crew_power_total", "avg_production_power", "avg_production_power_total",
        "cast_len", "crew_len", "production_companies_len", "production_countries_len", "keywords_len",
        "genres_len", "original_title_letter_count", "original_title_word_count", "title_word_count",
        "overview_word_count", "tagline_word_count", "log_popularity", "log_budget", "release_year", 
        "log_avg_star_power", "log_avg_star_power_total", "log_avg_crew_power", "log_avg_crew_power_total", 
        "log_avg_production_power", "log_avg_production_power_total"]])

df_tmdb.head(2)
#delete final columns 
for columns in ["id", "release_date", "release_month", "release_day", "release_quarter","release_dayofweek", "spoken_langauages"]:
    df_tmdb.drop(columns, axis=1, inplace=True)
df_tmdb.to_csv('tmdb.csv', index=False)
#end of ETLand feature engineering. Split data back into train and test set 

df_train= df_tmdb.iloc[0:3000] # first 3000 rows of the tmdb dataframe
df_test=df_tmdb.iloc[3000:7398]


#78 features are used to predict box office (log)revenue
df_train.shape, df_test.shape
# look at correlation of most important feature to discover multicollinearity
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,18))
sns.heatmap(df_train[['log_revenue','log_budget','budget', 'popularity', 'log_popularity', 'avg_star_power',
          'avg_crew_power','avg_production_power', 'log_avg_star_power', 'log_avg_crew_power', 'log_avg_production_power', 
          'production_USA', 'homepage', 'belongs_to_collection', 'release_year']].corr(), annot=True, fmt='.2', center=0.0, cmap='coolwarm')

#in general, linear or regression models suffer from features that are highly correlated. Since non of the features correlate more than (-)0.5 with each other, all features will be used
#split data into features and target. All features are either integer, float or dummy.
features = df_train.select_dtypes(include=['int64', 'float64', 'uint8', 'int8']).columns.tolist()
features.remove('log_revenue')
features_unseen = df_test.select_dtypes(include=['int64', 'float64', 'uint8', 'int8']).columns.tolist()
features_unseen.remove('log_revenue')

X, y = df_train[features], df_train['log_revenue']

#build simple lineair regression model. 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
#make predictions
y_pred=model.predict(X)
#check metrics, R_Squared is 0.58, rmse is 1,61

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

rmse = sqrt(mean_squared_error(y, y_pred))


print ("R-Squared is:", metrics.r2_score(y, y_pred))
print ("The rmse is:", rmse)
## The line / model

import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(y, y_pred)
#run linear regression model with test set 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


lmA = linear_model.LinearRegression()

#train model using training set 
lmA.fit(X_train, y_train)


#make predictions using the test set 
y_predA = lmA.predict(X_test)
## The line / model

import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(y_test, y_predA)
#rmse and R-Squared down to 0.48, RMSE up to 2.32
rmse = sqrt(mean_squared_error(y_test, y_predA))


print ("R-Squared is:", metrics.r2_score(y_test, y_predA))
print ("The rmse is:", rmse)
#look at actual and predicted values 
compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_predA})
compare.head(5)
# look at actual and predicted values of first 50 entries in the dataset
compare1 = compare.head(50)
compare1.plot(kind='bar',figsize=(30,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#make prediction on unseen test dataset
X_unseen=df_test[features_unseen]
prediction_unseen= lmA.predict(X_unseen)

prediction_unseen
df_submission['revenue'] = np.expm1(prediction_unseen)

df_submission.head(5)
df_submission[['id','revenue']].to_csv('submission_linear.csv', index=False)
# Random Forest Regressor model

# Import the model
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(X, y);
#make prediction
y_pred2=rf.predict(X)
#check metrics, R-Squared 0.93, RMSE 0.79
rmse = sqrt(mean_squared_error(y, y_pred2))


print ("R-Squared is:", metrics.r2_score(y, y_pred2))
print ("The rmse is:", rmse)
#run Random Forrest Regressor model with test set

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.3)
print (X2_train.shape, y2_train.shape)
print (X2_test.shape, y2_test.shape)

# fit a model
# Import the model
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf_test = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf_test.fit(X2_train, y2_train);
#make prediction on test set 
y_predC=rf_test.predict(X2_test)
#rmse and R-Squared down to 0.51, RMSE up to 2.23
rmse = sqrt(mean_squared_error(y2_test, y_predC))


print ("R-Squared is:", metrics.r2_score(y2_test, y_predC))
print ("The rmse is:", rmse)
#look at actual and predicted values 
compare = pd.DataFrame({'Actual': y2_test, 'Predicted': y_predC})
compare.head(5)
# look at actual and predicted values of first 50 entries in the dataset
compare1 = compare.head(50)
compare1.plot(kind='bar',figsize=(30,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#make prediction on unseen test dataset
X_unseen2=df_test[features_unseen]
prediction_unseen2= rf_test.predict(X_unseen2)

df_submission['revenue'] = np.expm1(prediction_unseen2)

df_submission.head(5)
df_submission[['id','revenue']].to_csv('submission_rf.csv', index=False)