# importing important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model  # will be using for plotting trend line
from sklearn.preprocessing import MinMaxScaler # for normalizing data
from sklearn.cluster import KMeans 
%matplotlib inline
# importing data
spotify = pd.read_csv('../input/spotify-dataset-19212020-160k-tracks/data.csv').drop(columns='Unnamed: 0')
# 10 random rows
spotify.sample(5)
# removing waste stuff(square bracket and quotation marks) from artist's name 
spotify['artists'] = spotify['artists'].apply(lambda x: x[1:-1].replace("'", ''))
# correcting data types 
spotify['release_date'] = pd.to_datetime(spotify['release_date'])
# spotify['year'] = pd.to_datetime(spotify['year'].apply(lambda x: str(x)+'-01-01'))
# finding correlation
corr = spotify.corr()
# visualizing correlaiton with heatmap
plt.figure(figsize=(20,8))
sns.heatmap(corr, vmax=1, vmin=-1, center=0,linewidth=.5,square=True, annot = True, annot_kws = {'size':8},fmt='.1f', cmap='BrBG_r')
plt.title('Correlation')
plt.show()
# lets see top10 artists who sang more songs than others
Top10 = pd.DataFrame({'No of songs':spotify['artists'].value_counts().head(10)})
Top10.plot.bar(color='brown')
plt.title('Top 10 artists')
plt.xlabel('Artists')
plt.ylabel('No of song')
plt.show()
# lets analyze artists whose songs are too popular
# spotify[['artists', 'name', 'popularity']].sort_values(by=['popularity'], ascending =False)
artists_grp = spotify.groupby(['artists'])
Top20artists = artists_grp[['popularity']].sum().sort_values(by=['popularity'], ascending=False)[:20]
Top20artists.plot.barh(color='orange')
plt.title('Artists Popularity')
plt.xlabel('Popularity')
plt.ylabel('Artists')
plt.show()
# lets analyze the popularity of The Beatles songs over the year
Beatles = spotify[spotify['artists'] == 'The Beatles']
# grid
sns.set(style='darkgrid')
# line plot passing x,y
sns.lineplot(x='release_date', y='popularity',lw = 1, data=Beatles, color='blue')
# Labels
plt.title("The Beatles Popularity")
plt.xlabel('Year')
plt.ylabel('Popularity')
plt.show()
# so now lets analyze which features of songs is affecting popularity in Beatles songs
plt.figure(0, figsize=(24,10))
x_axis = ['acousticness','danceability', 'duration_ms', 'energy', 'instrumentalness',
          'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
z = 0
for i in range(2):
    for j in range(6):
        # values to be plotted on axis(x,y)
        x = Beatles[x_axis[z]].values.reshape(-1,1)
        y = Beatles["popularity"].values.reshape(-1,1)
        # linear model 
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        # sub-plot
        ax = plt.subplot2grid((2,6), (i,j))
        ax.scatter(x,y, c='purple')
        # adding trend line
        ax.plot(x, regr.predict(x), color="red", linewidth=2, linestyle='--')
        # adding title
        ax.title.set_text(f'{x_axis[z]} vs popularity')
        z += 1
plt.show()
# lets analyze the increasing listener over the year
year_grp = spotify.groupby(['year'], as_index=False)
popularity_track = year_grp[['name', 'popularity']].sum()
song_count = year_grp['name'].count()

# dual axis graph
fig, ax1 = plt.subplots()
# Popularity of songs
ax1.plot(popularity_track.year, popularity_track.popularity, color='skyblue', linewidth=3,)
ax1.set_title('No of Song Vs Popularity Over the years', fontsize=14)
ax1.set_xlabel('Year', fontsize=18)
ax1.set_ylabel('Popularity', color='skyblue', fontsize=18)
# Adding second axis to the graph
ax2 = ax1.twinx()
# No of songs 
ax2.plot(song_count.year, song_count.name, color='pink', linewidth=3)
ax2.set_ylabel('Total song', color='pink', fontsize=18)
fig.tight_layout()
plt.show()
# lets see the increase of artists over the year
artists_data = {}
# Avoid duplicates
added = []
for year in spotify['year'].unique():
    # temporary container 
    total_artist = []
    for artist in spotify[spotify.year == year]['artists'].unique():
        if artist not in added:
            total_artist.append(artist)
    artists_data[year] = len(total_artist)
    added.extend(total_artist)
# creating dataframe 
artists_record = pd.DataFrame({'Year': list(artists_data.keys()),
                              'Increased_artists': list(artists_data.values())})
# line plot 
sns.lineplot('Year', 'Increased_artists', color='maroon', data=artists_record)
plt.title('Increasing Artists')
plt.xlabel('Year')
plt.ylabel('Increased Artists')
plt.show()
# lets perform clustering
# data(columns) we will we using
song_features = pd.DataFrame()
# normalizer instance
scaler = MinMaxScaler()
for col in spotify.iloc[:,:-1].columns:      # excluding year col i.e, of int64 type
    if spotify[col].dtypes in ['float64', 'int64']:
        # adding normalized col
        scaler.fit(spotify[[col]])
        song_features[col] = scaler.transform(spotify[col].values.reshape(-1,1)).ravel()     
# first we would like to know that how many cluster or to say Genres can be clustered 
# with less SSE(Sum of Squared Error) we will use "Elbow method" to find out 

# KMeans instance
km = KMeans()
k_rng = range(1,200)  # k value
sse = [] # sse value for each k
for i in k_rng:
    km = KMeans(n_clusters = i)
    km.fit(song_features.sample(1000))
    # calculating sse
    sse.append(km.inertia_) 
    
# due to less computation power I am unable to use whole data 
# I guess 1000 sample of whole data can depict actual
plt.plot(k_rng,sse)
plt.xlabel('K value')
plt.ylabel('SSE Error')
plt.title('Best K value')
# plt.ylim(0,400)
# plt.xlim(0,100)
plt.show()
# looks like 25 is good value of K
km = KMeans(n_clusters=25)
predicted_genres = km.fit_predict(song_features)
song_features['predicted_genres'] = predicted_genres
song_features['predicted_genres'] = song_features['predicted_genres'].apply(lambda x: 'Genre'+ str(x))
song_features.sample(10)
# lets see how many songs falls in each Genre and which Genre have more songs
genres_grp = song_features.groupby(['predicted_genres']).size()
plt.figure(figsize=(10,6))
genres_grp.sort_values(ascending=True).plot.barh(color='red')
plt.xlabel('Total Songs')
plt.title('Genre Ranking')
plt.show()
# reading artists data
artists_df = pd.read_csv('../input/spotify-dataset-19212020-160k-tracks/data_by_artist.csv')
artists_df = artists_df.rename(columns={"count": "playCount"})
# we will replace each feature with its Genre for our convience and for easy tracking
artists_df.iloc[:,1:-1] = scaler.fit_transform(artists_df.iloc[:,1:-1])
km = KMeans(n_clusters=25)
artists_df['genres'] = km.fit_predict(artists_df.iloc[:,1:-1])
artists_df = artists_df.iloc[:,[0,-3,-2,-1]]
artists_df.head()
# lets create our own user list with his rating and add to artists data
artists_df['user_id'] = np.random.randint(1000,1400,len(artists_df))
artists_df['rating'] = np.random.randint(1,6,len(artists_df))
artists_df.head()
# lets create our recommender system
def recommend_me(user):
    """This function will recommend artists to any user with its genre profile"""
    # first we will choose user top liked genres
    fav_genre = artists_df[artists_df['user_id']==user].sort_values(by=['rating','playCount'], ascending=False)['genres'][:5]
    fav_genre = list(dict.fromkeys(fav_genre)) # removing duplicate if exits
    
    # lets clear out the artists from list whose songs has been listened by the user
    listened_artist = artists_df.index[artists_df['artists'].isin(['Johann Sebastian Bach','Frédéric Chopin'])].tolist()
    
    # rest data
    remaining_artist = artists_df.drop(listened_artist, axis=0)
    CanBeRecommened =  remaining_artist[remaining_artist['genres'].isin(fav_genre)]
    
    # now lets sort our artists whose are popular in this user favorite genre
    CanBeRecommened = CanBeRecommened.sort_values(by=['rating','playCount',], ascending=False)[['artists', 'genres', 'rating', 'playCount']][:5]
    
    # output will contain artists name, genres, other useres rating and song played count
    return CanBeRecommened
# lets recommend this user some artists
recommend_me(1012)
# lets check which genre is user fav and did he get same recommended
artists_df[artists_df.user_id==1012].sort_values(by='rating')['genres'].unique()
# here we can see that user fav genres include 13,7 and 14 and we recommended that too