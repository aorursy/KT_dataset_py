import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
%matplotlib inline
sns.set_style('whitegrid')
def load_credits_data(file_path):
    df = pd.read_csv(file_path, dtype='unicode')
    # all json columns`
    json_columns = ['cast', 'crew']
    for column in json_columns:
        # use ast because json data has single quotes in the csv, which is invalid for a json object; it should be " normally
        df[column] = df[column].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))
    return df
credits = load_credits_data(r"../input/credits.csv")
credits.shape
credits.columns
credits.info()
credits.isnull().sum()
credits_flattened = pd.DataFrame(None,None,columns=['movie_id','actor_1_gender','actor_2_gender','actor_3_gender',\
                        'actor_4_gender','actor_5_gender','actor_1_name','actor_2_name','actor_3_name','actor_4_name',\
                        'actor_5_name','director_gender','director_name','producer_gender','producer_name','casting_gender',\
                                'casting_name'])

for i,row in credits.iterrows():
    # dummy row
    newrow = {'movie_id':np.nan,'actor_1_gender':np.nan,'actor_2_gender':np.nan,'actor_3_gender':np.nan,'actor_4_gender':np.nan,\
              'actor_5_gender':np.nan,'actor_1_name':np.nan,'actor_2_name':np.nan,'actor_3_name':np.nan,'actor_4_name':np.nan,\
            'actor_5_name':np.nan,'director_gender':np.nan,'director_name':np.nan,'producer_gender':np.nan,\
            'producer_name':np.nan,'casting_gender':np.nan,'casting_name':np.nan}
    
    # fill movie id
    newrow['movie_id'] = int(row['id'])
    
    # fill cast
    count=1
    for item in row['cast']:
        if count==6:
            break
        if 'gender' in item and 'name' in item:
            newrow['actor_'+str(count)+'_gender'] = item['gender']
            newrow['actor_'+str(count)+'_name'] = item['name']
            count += 1

    # fill crew
    director=0
    producer=0
    casting=0
    for item in row['crew']:
        if director and producer and casting:
            break
        if 'job' in item and item['job'] in ['Director','Producer','Casting'] and 'gender' in item and 'name' in item:
            if item['job'] == 'Director' and not director:
                newrow['director_gender'] = item['gender']
                newrow['director_name'] = item['name']
                director =  1
            elif item['job'] == 'Producer' and not producer:
                newrow['producer_gender'] = item['gender']
                newrow['producer_name'] = item['name']
                producer =  1
            elif item['job'] == 'Casting' and not casting:
                newrow['casting_gender'] = item['gender']
                newrow['casting_name'] = item['name']
                casting =  1

    credits_flattened = credits_flattened.append(newrow,ignore_index=True)
credits_flattened.shape
credits_flattened.head()
#credits_flattened.to_csv('flattened.csv',index=False)
def load_movies_metadata(file_path):
    df = pd.read_csv(file_path, dtype='unicode')
    # covert each item of release_date to datetime.date type entity
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: x.date())
    # all json columns`
    json_columns = ['belongs_to_collection', 'genres', 'production_companies', 'production_countries', 'spoken_languages']
    for column in json_columns:
        # use ast because json data has single quotes in the csv, which is invalid for a json object; it should be " normally
        df[column] = df[column].apply(lambda x: np.nan if pd.isnull(x) else ast.literal_eval(x))
    return df
movies = load_movies_metadata(r"../input/movies_metadata.csv")
movies.head(3)
movies.columns
movies_required = movies[['budget', 'genres', 'id', 'imdb_id', 'original_language', 'title', 'popularity', 'release_date', 'revenue','vote_average','vote_count']]
movies_required.head()
movies_required.shape
for i,row in movies_required.iterrows():
    if len(row[1])>0:
        if 'name' in row[1][0]:
            row[1] = row[1][0]['name']
movies_required.head()
movies_required.rename(columns={'genres':'genre_top','id':'movie_id'}, inplace=True)
movies_required.head()
movies_required.drop(movies_required.index[19730],inplace=True)
movies_required.drop(movies_required.index[29502],inplace=True)
movies_required.drop(movies_required.index[35585],inplace=True)
movies_required['movie_id'] = pd.to_numeric(movies_required['movie_id'])
movies_required.head()
merged_df = pd.merge(credits_flattened,movies_required,on=['movie_id'])
merged_df.shape
merged_df.columns
merged_df.head()
#merged_df.to_csv('flattened_merged.csv',index=False)