import numpy as np 
import pandas as pd 
raw_data_with_measures = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv')
raw_data_with_measures.head(4)
# lets see if there is missing data

def display_missing(df):    
    for col in df.columns.tolist():   
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))   
    print('\n')
    
display_missing(raw_data_with_measures)
raw_data_with_measures['artists'] = raw_data_with_measures['artists'].apply(lambda x: x[1:-1].split(', ')) # makes the string lists in artists column into actual lists

raw_data_with_measures = raw_data_with_measures.explode('artists') # opens the lists so that each artist is on a song has his own row with the song

raw_data_with_measures['artists'] = raw_data_with_measures['artists'].apply(lambda x: x.strip("'")) #  we want to take off the quotes on the artists

raw_data =  raw_data_with_measures.loc[:,['artists','name','popularity','year']] # we will just focus on these columns for now, and rename our data to raw_data


raw_data
raw_data['artists'].value_counts()
import plotly.express as px

def top_n_artists_by_song_count(data, lower_bound, upper_bound):
    
    reindex_order = data['artists'].value_counts()[lower_bound:upper_bound].index[::-1]     
    # order of index to make right values in right places
        
    total_value_of_songs = data['artists'].value_counts()[lower_bound:upper_bound].reindex(reindex_order)
    
    total_value_of_popularity =  data.groupby(['artists']).sum()['popularity'].reindex(reindex_order)   # so we can see their popularity too
    
    df = pd.DataFrame({('Artist ') :total_value_of_songs.index, 
                        ('Total Song Count '): total_value_of_songs.values,
                        ('Total Artist Popularity '): total_value_of_popularity.values}) 
  
    fig = px.bar(df, x = ('Total Song Count ') , y = ('Artist '),
                    
                     hover_data = [('Total Song Count '), ('Total Artist Popularity ')],
                                       
                     color = ('Total Song Count ') , title= f'Artists Song Count Ranked ({lower_bound+1},{upper_bound}) ',
               
                     height = 700  )

    return fig.show()
top_n_artists_by_song_count(raw_data,0,20)
raw_data.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(20)
def top_n_artists_by_popularity(data, lower_bound, upper_bound):
    

    reindex_order =  data.groupby(['artists']).sum()['popularity'].sort_values(ascending = False)[lower_bound:upper_bound].index[::-1]     
    # order of index to make right values in right places
        
    total_value_of_popularity =  data.groupby(['artists']).sum()['popularity'].sort_values(ascending = False)[
        lower_bound:upper_bound].reindex(reindex_order)
    
    total_songs = data['artists'].value_counts().round(0).reindex(reindex_order)   
    
    df = pd.DataFrame({('Artist ') :total_value_of_popularity.index, 
                        ('Total Artist Popularity '): total_value_of_popularity.values,
                        ('Total Song Count '): total_songs.values}) 
  
    fig = px.bar(df, x = ('Total Artist Popularity ') , y = ('Artist '),
                    
                     hover_data = [('Total Artist Popularity '), ('Total Song Count ')],
                                       
                     color = ('Total Artist Popularity ') , title= f'Artists Popularity Ranked ({lower_bound +1},{upper_bound})',
               
                     height = 700  )

    return fig.show()
top_n_artists_by_popularity(raw_data, 0, 20)
# most popular by year

pop_year = raw_data.sort_values('popularity', ascending=False).groupby('year').first()
pop_year = pop_year.reset_index()
pop_year = pop_year[['year', 'artists', 'name', 'popularity']]

pop_year
def top_songs_by_year(data, lower_bound, upper_bound):
    
    pop_year = data.sort_values('popularity', ascending=False).groupby('year').first()
    pop_year = pop_year.reset_index()
    pop_year = pop_year[['year', 'artists', 'name', 'popularity']]

    reindex_order =  pop_year[lower_bound -1921  :upper_bound -1921].index[::-1]     
    # order of index to make right values in right places
        
    total_song_popularity =  pop_year[lower_bound -1921 :upper_bound -1921].reindex(reindex_order)
    
    
    df = pd.DataFrame({('Year ') :total_song_popularity['year'].values, 
                        ('Top Song Popularity '): total_song_popularity['popularity'].values,
                        ('Artist '): total_song_popularity['artists'].values,
                        ('Song '): total_song_popularity['name'].values })
  
    fig = px.line(df, x = ('Year ') , y = ('Top Song Popularity '),
                    
                     hover_data = [('Top Song Popularity '), ('Year '), ('Artist '), ('Song ') ],
                  
                     
                                       
                     title= f'Top Songs by Year ({lower_bound } - {upper_bound})', 
               
                     height = 700  )
    
    fig.update_traces(mode='markers+lines')

    return fig.show()
top_songs_by_year(raw_data, 1921,2020)
# lets make a decades column to further sort our list

raw_data['Song Decade'] = None

raw_data.loc[(raw_data['year'] >= 1920) & (raw_data['year'] < 1930), 'Song Decade'] = '1920s'
raw_data.loc[(raw_data['year'] >= 1930) & (raw_data['year'] < 1940), 'Song Decade'] = '1930s'
raw_data.loc[(raw_data['year'] >= 1940) & (raw_data['year'] < 1950), 'Song Decade'] = '1940s'
raw_data.loc[(raw_data['year'] >= 1950) & (raw_data['year'] < 1960), 'Song Decade'] = '1950s'
raw_data.loc[(raw_data['year'] >= 1960) & (raw_data['year'] < 1970), 'Song Decade'] = '1960s'
raw_data.loc[(raw_data['year'] >= 1970) & (raw_data['year'] < 1980) , 'Song Decade'] = '1970s'
raw_data.loc[(raw_data['year'] >= 1980) & (raw_data['year'] < 1990) , 'Song Decade'] = '1980s'
raw_data.loc[(raw_data['year'] >= 1990) & (raw_data['year'] < 2000) , 'Song Decade'] = '1990s'
raw_data.loc[(raw_data['year'] >= 2000) & (raw_data['year'] < 2010) , 'Song Decade'] = '2000s'
raw_data.loc[(raw_data['year'] >= 2010) & (raw_data['year'] < 2020) , 'Song Decade'] = '2010s'
raw_data.loc[(raw_data['year'] >= 2020) & (raw_data['year'] < 2030) , 'Song Decade'] = '2020s'

raw_data
most_popular_song_decade = raw_data.sort_values('popularity', ascending=False).groupby('Song Decade').first()
most_popular_song_decade = most_popular_song_decade.reset_index()
most_popular_song_decade = most_popular_song_decade[['Song Decade', 'artists', 'name', 'popularity']]

most_popular_song_decade
most_pop_decade = raw_data.groupby(["artists","Song Decade"])["popularity"].sum()
most_pop_decade = most_pop_decade.reset_index()
most_pop_decade['Decade Song Count'] = raw_data.groupby(["artists","Song Decade"])['artists'].value_counts().values
most_pop_decade = most_pop_decade.sort_values(["Song Decade","popularity"], ascending = False)
most_pop_decade.groupby('Song Decade').first()
def top_artists_by_decade(data, decade, lower_bound, upper_bound): 
    
    most_pop_decade = raw_data.groupby(["artists","Song Decade"])["popularity"].sum()
    most_pop_decade = most_pop_decade.reset_index()
    most_pop_decade['Decade Song Count'] = raw_data.groupby(["artists","Song Decade"])['artists'].value_counts().values
    most_pop_decade = most_pop_decade.sort_values(["Song Decade","popularity"], ascending = False)
    most_pop_decade = most_pop_decade[most_pop_decade['Song Decade'] == decade]
    
    reindex_order = most_pop_decade[lower_bound:upper_bound].index[::-1]     
    # order of index to make right values in right places
        
    total_value_of_popularity =  most_pop_decade[lower_bound:upper_bound].reindex(reindex_order)
    
    df = pd.DataFrame({('Artist ') :total_value_of_popularity['artists'].values, 
                        ('Total Decade Popularity '): total_value_of_popularity['popularity'].values,
                        ('Decade Song Count '): total_value_of_popularity['Decade Song Count'].values}) 
  
    fig = px.bar(df, x = ('Total Decade Popularity ') , y = ('Artist '),
                    
                     hover_data = [('Total Decade Popularity '), ('Artist '), ('Decade Song Count ')],
                                       
                     color = ('Total Decade Popularity ') , title= f'Most Popular Artists {decade} Ranked ({lower_bound +1},{upper_bound})',
               
                     height = 700  )

    return fig.show()
top_artists_by_decade(raw_data, '2000s', 0, 20) 
song_count_decade = raw_data['Song Decade'].value_counts()
song_count_decade = song_count_decade.reset_index()
song_count_decade.columns = ['Decade', 'Song Count']
song_count_decade = song_count_decade.sort_values(by= 'Decade')
song_count_decade
import plotly.express as px

fig = px.pie(song_count_decade, values= song_count_decade['Song Count'] , names= song_count_decade['Decade'], title='Songs Released in Each Decade')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
raw_data_genre = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv')
raw_data_genre.head(4)
# lets see if there is missing data

def display_missing(df):    
    for col in df.columns.tolist():   
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))   
    print('\n')
    
display_missing(raw_data)
pd.options.mode.chained_assignment = None 

data_genre = raw_data_genre.loc[:,['artists','count','genres','popularity']] # lets explore these columns for now
data_genre['genres'] = data_genre['genres'].apply(lambda x: x[1:-1].split(', '))  # makes it into actual list instead of string list

for i in data_genre.index:

    data_genre['genres'].loc[i] = list(filter(None, data_genre['genres'][i]))  # filters out empty values in each list
    

data_genre.head(10)
count_genres = data_genre['genres'].explode().value_counts()
count_genres
def top_genres(data, lower_bound, upper_bound): 
    
    count_genres = data_genre['genres'].explode().value_counts()    
    
    reindex_order = count_genres[lower_bound:upper_bound].index[::-1]     
    # order of index to make right values in right places
        
    count_genres_ranked =  count_genres[lower_bound:upper_bound].reindex(reindex_order)
    
    df = pd.DataFrame({('Genre ') :count_genres_ranked.index, 
                        ('Artist Count '): count_genres_ranked.values}) 
  
    fig = px.bar(df, x = ('Artist Count ' ) , y = ('Genre '),
                    
                     hover_data = [('Artist Count ' ) , ('Genre ')],
                                       
                     color = ('Artist Count ') , title= f'Most Popular Genres Ranked ({lower_bound +1},{upper_bound})',
               
                     height = 700  )

    return fig.show()
top_genres(data_genre, 0, 20)
genre_counts = raw_data_genre.groupby('genres')['artists'].count().sort_values(ascending=False)  
genre_counts.head(10)
missing_genre_artists = data_genre[data_genre['genres'].map(lambda x: len(x)) < 1]
missing_genre_artists
missing_genre_artists_split = missing_genre_artists.copy()
missing_genre_artists_split['artists'] = missing_genre_artists_split['artists'].apply(lambda x: x.split())  # split into words
missing_genre_artists_split
# lets find most common words in artists

def word_count(data):

    all_words = []  
    for elmnt in data['artists']:  
        all_words += elmnt  

    val_counts = pd.Series(all_words).value_counts()

    return val_counts.head(40)

word_count(missing_genre_artists_split)
data_genre[data_genre['genres'].map(lambda x: len(x)) < 1].sort_values(ascending = False,by = 'count').head(20)
pd.options.mode.chained_assignment = None 

for i in data_genre.index:
    
    if 'Broadway' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'broadway'")
    if 'Motion Picture' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'movie tunes'")
    if 'Cast' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'ensemble'")     # we will put any cast artists under ensemble genre 
    if 'Ensemble' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'ensemble'")  
    if 'Orchestra' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'orchestra'")
    if 'Orquesta' in data_genre['artists'].loc[i]:              # these are just vara
        data_genre['genres'].loc[i].append("'orchestra'")
    if 'Orchestre' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'orchestra'")
    if 'Chorus' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'chorus'")
    if 'Choir' in data_genre['artists'].loc[i]:
        data_genre['genres'].loc[i].append("'choir'")
        
    data_genre['genres'].loc[i] = list(dict.fromkeys(data_genre['genres'].loc[i])) # removes duplicates
    

data_genre.head(12)
final_data_genre = data_genre[data_genre['genres'].map(lambda x: len(x)) > 0] # removes empty lists
final_data_genre = final_data_genre.reset_index(drop=True)
final_data_genre = final_data_genre.drop('count', axis =1 )
final_data_genre.rename(columns={'popularity': 'Artist Popularity'}, inplace=True)
final_data_genre['Artist Popularity'] = final_data_genre['Artist Popularity'].astype(int)
final_data_genre
def rank_artist_similarity(data, artist, genre_parameter):
    artist_data = data[data.artists == artist]
    artist_genres = set(*artist_data.genres)
    similarity_data = data.drop(artist_data.index)
    similarity_data.genres = similarity_data.genres.apply(lambda genres: list(set(genres).intersection(artist_genres)))
    similarity_lengths = similarity_data.genres.str.len()
    similarity_data = similarity_data.reindex(similarity_lengths[similarity_lengths >= genre_parameter].sort_values(ascending=False).index)
    similarity_data.rename(columns={'artists': f'Similar Artists to {artist}', 'genres': 'Similar Genres', 'popularity': 'Artist Popularity'}, inplace=True)
    return similarity_data
rank_artist_similarity(final_data_genre, 'Eminem',3)
rank_artist_similarity(final_data_genre, 'Foo Fighters',6)
rank_artist_similarity(final_data_genre, 'Taylor Swift',3)
merged_df = raw_data.merge(final_data_genre, how = 'inner', on = ['artists'])
merged_df_copy = merged_df.copy()
merged_df_copy.rename(columns={'artists': 'Artist', 'name':'Song Name','popularity':'Song Popularity','year':'Year','genres':'Genres'}, inplace=True)

merged_df_copy
def rank_song_similarity(data, song, artist, genre_parameter):
    
    song_and_artist_data = data[(data.Artist == artist) & (data['Song Name'] == song)].sort_values('Year')[0:1]  # this ensures the first song is picked, not any remasters
    artist_genres = set(*song_and_artist_data.Genres)

    similarity_data = data[~data.Artist.str.contains(artist)] # drops the artist from the dataframe
    
    similarity_data.Genres = similarity_data.Genres.apply(lambda Genres: list(set(Genres).intersection(artist_genres)))
    
    similarity_lengths = similarity_data.Genres.str.len()
    similarity_data = similarity_data.reindex(similarity_lengths[similarity_lengths >= genre_parameter].sort_values(ascending=False).index)
    
    similarity_data = similarity_data[similarity_data['Song Decade'] == song_and_artist_data['Song Decade'].values[0]]
    
    similarity_data = similarity_data.sort_values(by ='Song Popularity', ascending = False)
    
    
    similarity_data.rename(columns={'Song Name': f'Similar Song to {song}', 'Genres' : 'Similar Genres'}, inplace=True)
    return similarity_data.head(30)

rank_song_similarity(merged_df_copy, 'Bohemian Rhapsody', 'Queen',2)
rank_song_similarity(merged_df_copy, 'Learn to Fly', 'Foo Fighters',4)
rank_song_similarity(merged_df_copy, 'Learn to Fly', 'Foo Fighters',3)
rank_song_similarity(merged_df_copy, 'Without Me', 'Eminem',2)
rank_song_similarity(merged_df_copy, 'Without Me', 'Eminem',3)

merged_df_with_measures = raw_data_with_measures.merge(final_data_genre, how = 'inner', on = ['artists'])
merged_df_with_measures.rename(columns={'artists': 'Artist', 'name':'Song Name','popularity':'Song Popularity','year':'Year','genres':'Genres'}, inplace=True)
merged_df_with_measures = merged_df_with_measures.drop(['duration_ms','explicit','id','release_date'], axis =1 )

merged_df_with_measures
# this is missing the song deacades, lets put it in again

merged_df_with_measures['Song Decade'] = None

merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1920) & (merged_df_with_measures['Year'] < 1930), 'Song Decade'] = '1920s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1930) & (merged_df_with_measures['Year'] < 1940), 'Song Decade'] = '1930s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1940) & (merged_df_with_measures['Year'] < 1950), 'Song Decade'] = '1940s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1950) & (merged_df_with_measures['Year'] < 1960), 'Song Decade'] = '1950s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1960) & (merged_df_with_measures['Year'] < 1970), 'Song Decade'] = '1960s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1970) & (merged_df_with_measures['Year'] < 1980) , 'Song Decade'] = '1970s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1980) & (merged_df_with_measures['Year'] < 1990) , 'Song Decade'] = '1980s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 1990) & (merged_df_with_measures['Year'] < 2000) , 'Song Decade'] = '1990s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 2000) & (merged_df_with_measures['Year'] < 2010) , 'Song Decade'] = '2000s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 2010) & (merged_df_with_measures['Year'] < 2020) , 'Song Decade'] = '2010s'
merged_df_with_measures.loc[(merged_df_with_measures['Year'] >= 2020) & (merged_df_with_measures['Year'] < 2030) , 'Song Decade'] = '2020s'

merged_df_with_measures
columns_reorder = ['Artist', 'Song Name', 'Song Popularity','Year','Genres','Artist Popularity', 'Song Decade', 'acousticness', 
                   'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']

merged_df_with_measures = merged_df_with_measures.reindex(columns=columns_reorder)

merged_df_with_measures
song_data= merged_df_with_measures.loc[:,['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

song_features = pd.DataFrame()

for col in song_data.iloc[:,:].columns:    
       
    scaler.fit(song_data[[col]])
    song_features[col] = scaler.transform(song_data[col].values.reshape(-1,1)).ravel() 
song_features
data_to_merge = merged_df_with_measures.drop(['acousticness', 'danceability',
       'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'valence'], axis = 1)

final_merged_df = data_to_merge.join(song_features)
final_merged_df
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def rank_song_similarity_by_measure(data, song, artist, genre_parameter):
    
    song_and_artist_data = data[(data.Artist == artist) & (data['Song Name'] == song)].sort_values('Year')[0:1]
    
    similarity_data = data.copy()
    
    data_values = similarity_data.loc[:,['acousticness', 'danceability',
       'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
       'speechiness', 'tempo', 'valence']]
    
    similarity_data['Similarity with song'] =cosine_similarity(data_values, data_values.to_numpy()[song_and_artist_data.index[0],None]).squeeze()
    
    artist_genres = set(*song_and_artist_data.Genres)

    similarity_data.Genres = similarity_data.Genres.apply(lambda Genres: list(set(Genres).intersection(artist_genres)))
    
    similarity_lengths = similarity_data.Genres.str.len()
    similarity_data = similarity_data.reindex(similarity_lengths[similarity_lengths >= genre_parameter].sort_values(ascending=False).index)
    
    similarity_data = similarity_data[similarity_data['Song Decade'] == song_and_artist_data['Song Decade'].values[0]]
 
    similarity_data.rename(columns={'Song Name': f'Similar Song to {song}'}, inplace=True)
    
    similarity_data = similarity_data.sort_values(by= 'Similarity with song', ascending = False)
    
    similarity_data = similarity_data[['Artist', f'Similar Song to {song}',
       'Song Popularity', 'Year', 'Genres', 'Artist Popularity', 'Song Decade', 'Similarity with song',
       'acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]
    
    return similarity_data.head(15)
rank_song_similarity_by_measure(final_merged_df, 'Bohemian Rhapsody', 'Queen',2)
rank_song_similarity_by_measure(final_merged_df, 'Learn to Fly', 'Foo Fighters', 4)