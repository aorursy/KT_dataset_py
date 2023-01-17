# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ast import literal_eval

data=pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv')

data.head()

data.shape

data=data.iloc[1:80000]

#data[data['artists']=='Slipknot']

data[data['name']=='Snuff']

data.columns

#data_genre['artists']=data_genre['artists'].apply(literal_eval)

data['artists']=data['artists'].apply(literal_eval)



#Recommend songs based on artists

song=data[['artists','name']]

song

def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

       

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''    

song['artists']=song['artists'].apply(clean_data)

song

def create_soup(x):

    return ''.join(x['artists'])+' '

song['soup']=song.apply(create_soup,axis=1)

song['soup']

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

cm=cv.fit_transform(song['soup'])

cm



from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(cm,cm)



name=song['name']



song=song.reset_index()



name_index=pd.Series(song.index,index=song['name'])



def recommendation(names):

    idx = name_index[names]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    song_indices = [i[0] for i in sim_scores]

    print('Similiar songs like',names)

    return name.iloc[song_indices]

recommendation('Snuff').head(10)

#Recommendation artist based on genres

data_genre=pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv')

data_genre['genres']=data_genre['genres'].apply(literal_eval)

data_genre.shape

genres=data_genre[['artists','genres']]

def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

       

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''  

genres['genres']=genres['genres'].apply(clean_data)

genres

def create_genres(x):

    return ' '.join(x['genres'])+' '

genres['soup']=genres.apply(create_genres,axis=1)

genres['soup']

cm1=cv.fit_transform(genres['soup'])

cosine_sim1=cosine_similarity(cm1,cm1)  

artist=genres['artists']

genres=genres.reset_index()

artist_index=pd.Series(genres.index,index=genres['artists'])

def recommendation_artist(names):

    idx = artist_index[names]

    sim_scores = list(enumerate(cosine_sim1[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    song_indices = [i[0] for i in sim_scores]

    print('Similiar artists like',names)

    return artist.iloc[song_indices]

recommendation_artist('Lata Mangeshkar').head(10)

#Recommend artist based on Song Features

data_genre=pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv')

data_genre.head()

data_genre.shape

data_genre.artists.str.strip()

data_genre=data_genre.drop_duplicates(subset='artists',keep=False)

data_genre=data_genre.drop(['key','mode','count','duration_ms','genres'],axis=1)

data_genre

data_genre['acousticness'].max()

data_genre['acousticness']=data_genre['acousticness']/0.996

data_genre['tempo'].max()

data_genre['tempo']=data_genre['tempo']/217.74

data_genre['loudness'].max()

data_genre['loudness']=data_genre['loudness']/1.342

data_genre['liveness'].max()

data_genre['liveness']=data_genre['liveness']/0.991

data_genre['danceability'].max()

data_genre['danceability']=data_genre['danceability']/0.986

data_genre['speechiness'].max()

data_genre['speechiness']=data_genre['speechiness']/0.96

data_genre['valence'].max()

data_genre['valence']=data_genre['valence']/0.991

data_genre['popularity'].max()

data_genre['popularity']=data_genre['popularity']/95

df2=pd.melt(data_genre,id_vars='artists',var_name='song_features',value_name='value')

df2

df_pivot=pd.pivot(df2,columns='artists',index='song_features',values='value')

df_pivot

#Pearson R Correlation

song_ratings=df_pivot['Aerosmith']

similiar_to_artists=df_pivot.corrwith(song_ratings)

artists_frame=pd.DataFrame(similiar_to_artists,columns=['PearsonR'])

artists_frame=artists_frame.reset_index()

artists_frame=artists_frame[artists_frame['PearsonR']!=1.000000]

#print('Recommendation for')

artists_frame.sort_values('PearsonR',ascending=False).head(15)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.listdir('../input')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session