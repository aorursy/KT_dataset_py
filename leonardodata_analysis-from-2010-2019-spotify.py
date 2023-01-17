import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

import folium

from folium.plugins import HeatMap
data = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='ISO-8859-1')

data.head()
data.shape
data.describe()
data.info()
data.count()
data = data.rename(columns={'top genre': 'top_genre'})

data = data.drop('Unnamed: 0', axis=1)

print(data.columns)
data.head(2)
data.corr()
plt.figure(figsize=(7, 6))

sn.heatmap(data.corr(),

            annot = True,

            fmt = '.2f',

            cmap='Blues')

plt.title('Correlation between variables in the Spotify data set')

plt.show()


data.plot(x='acous',y='nrgy',kind='scatter', title='Relationship between Energy and Acousticness  ',color='r')

plt.xlabel('Acousticness')

plt.ylabel('Energy')

data.plot(x='nrgy',y='dB',kind='scatter', title='Relationship between Loudness (dB) and Energy',color='b')

plt.xlabel('Energy')

plt.ylabel('Loudness (dB)')

data.plot(x='val',y='dnce',kind='scatter', title='Relationship between Loudness (dB) and Valence',color='g')

plt.xlabel('Valence')

plt.ylabel('Loudness (dB)')
artists = data['artist'].unique()

print("The dataset has {} artists".format (len(artists)))
artists = data['artist'].value_counts().reset_index().head(10)

print(artists)
plt.figure(figsize=(15,10))

sn.barplot(x='index',y='artist', data=artists)

plt.title("Number of Musics on top score by the 10 Top Artists")
frames = []

topArtists = data['artist'].value_counts().head(10).index

for i in topArtists:

     frames.append(data[data['artist'] == i])

        

resultArtist = pd.concat(frames)

artistsYear = pd.crosstab(resultArtist["artist"],resultArtist["year"],margins=False)

artistsYear
plt.figure(figsize=(20,10))

for i in artists['index']:

  tmp = []

  for y in range(2010,2020):

    songs = data[data['year'] == y][data['artist'] == i]

    tmp.append(songs.shape[0])

  sn.lineplot(x=list(range(2010,2020)),y=tmp)

plt.legend(list(artists['index']))

plt.title("Evolution of each Top 10 Artists throught the target Years")
data['artist'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%')
data[data['artist'] == 'Lady Gaga'][data['year'] == 2018]
data['title'].value_counts().head(20)>1
plt.figure(figsize=(15,10))

sn.countplot(y=data.title, order=pd.value_counts(data.title).iloc[:19].index, data=data)

topMusics = data['title'].value_counts().head(19).index

plt.title("The Songs appear more than once")



plt.figure(figsize=(20,10))

for i in topMusics:

  tmp = []

  for y in range(2010,2020):

    songs = data[data['year'] == y][data['title'] == i]

    tmp.append(songs.shape[0])

  sn.lineplot(x=list(range(2010,2020)),y=tmp)

plt.legend(list(topMusics))

plt.title("Evolution of each Top 10 Artists throught the target Years")
data[data['title']== 'Sugar']
data.sort_values(by=['pop'], ascending=False).head(15)
data.sort_values(by=['dur'], ascending=False).head(15)
data.sort_values(by=['acous'], ascending=False).head(15)
genres = data['top_genre'].value_counts().reset_index().head(10)
plt.figure(figsize=(23,10))

sn.barplot(x='index',y='top_genre', data=genres)
data['top_genre'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%')
plt.figure(figsize=(20,10))

for i in genres['index']:

  tmp = []

  for y in range(2010,2020):

    songs = data[data['year'] == y][data['top_genre'] == i]

    tmp.append(songs.shape[0])

  sn.lineplot(x=list(range(2010,2020)),y=tmp)

plt.legend(list(genres['index']))
artists
dicArtists = {

    'Katy Perry':"Santa Barbara",

    'Justin Bieber':"London Canada",

     'Rihanna':"Saint Michael",

    'Maroon 5':"Los Angeles",

    'Lady Gaga':"Manhattan",

    'Bruno Mars':"Honolulu", 

    'The Chainsmokers':"Times Square" ,

    'Pitbull':"Miami",

    'Shawn Mendes':"Toronto",

    'Ed Sheeran':"United Kingdom", 

  }
#!pip install folium

!pip install geocoder
import geocoder

listGeo = []



for value in (dicArtists.values()):

    g = geocoder.arcgis(value)

    listGeo.append(g.latlng)
top_genres =[]

for key in (dicArtists.keys()):

    top_genres.append(data[data['artist']== key].top_genre.unique())

lat = []

log = []

for i in listGeo:

    lat.append(i[0])

    log.append(i[1])
colors = {

 'dance pop': 'pink',

 'pop': 'blue',

 'barbadian pop': 'green',

 'electropop': 'orange',

 'canadian pop': 'red',

}

        
dfLocation = pd.DataFrame(columns=['Name','Lat','Log','Gen'])

dfLocation['Name'] = artists['index']

dfLocation['Gen']  = np.array(top_genres)

dfLocation['Lat']  = lat

dfLocation['Log']  = log

dfLocation
spotify = folium.Map(

    location=[41.5503200,-8.4200500],# Coordenadas retiradas do Google Maps

    zoom_start=2

)

spotify
for i in range(10):

    singer = dfLocation.iloc[i]

    folium.Marker(

        

        popup=singer['Name']+'-'+singer['Gen'],

        location=[singer['Lat'], singer['Log']],

    icon=folium.Icon(color=colors[singer['Gen']], icon='music')).add_to(spotify)

    

spotify
spotify = folium.Map(

    location=[41.5503200,-8.4200500],# Coordenadas retiradas do Google Maps

    zoom_start=2

)



HeatMap(list(zip(lat, log))).add_to(spotify)

spotify
dic = {

    'Katy Perry':"Santa Barbara",

    'Justin Bieber':"London Canada",

     'Rihanna':"Saint Michael",

    'Maroon 5':"Los Angeles",

    'Lady Gaga':"Manhattan",

    'Bruno Mars':"Honolulu", 

    'The Chainsmokers':"Times Square" ,

    'Pitbull':"Miami",

    'Shawn Mendes':"Toronto",

    'Ed Sheeran':"United Kingdom", 

    'Jennifer Lopez':'Castle Hill',

    'Calvin Harris' :  'Dumfries',  

    'Adele'  : 'Tottenham',

    'Kesha'     :  'California',

    'Justin Timberlake'   : 'Memphis' ,  

    'David Guetta '      :'Paris',

    'OneRepublic'       :'Colorado',

    'Britney Spears '    : 'Mississippi',

    'Ariana Grande '      :'Florida',

    'Taylor Swift'       :'Pennsylvania',  

  }

listGeo = []

for value in (dic.values()):

    g = geocoder.arcgis(value)

    listGeo.append(g.latlng)



lat = []

log = []

for i in listGeo:

    lat.append(i[0])

    log.append(i[1])



spotify = folium.Map(

    location=[41.5503200,-8.4200500],# Coordenadas retiradas do Google Maps

    zoom_start=2

)



HeatMap(list(zip(lat, log))).add_to(spotify)

spotify
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re, os

%config IPCompleter.greedy=True
len(data['top_genre'].unique())
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, LabelEncoder
labelenconder = LabelEncoder()

data['title'] = labelenconder.fit_transform(data['title'].astype('str'))

data['artist'] = labelenconder.fit_transform(data['artist'].astype('str'))

target = data['top_genre']
train = data.drop(columns=['top_genre'], axis=1) 

train.head()
Sum_of_squared_distances = []

std = StandardScaler()

std.fit(train)

data_transformed = std.transform(train)



K = range(1,60)

for i in K:

    km = KMeans(n_clusters=i)

    km.fit(data_transformed)

    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
km = KMeans(n_clusters=2)

km.fit(train)

km.cluster_centers_
def converter(cluster):

    result = re.findall(".*pop",cluster)

    if len(result) != 0:

        return 1

    else:

        return 0
data['is_pop'] = data['top_genre'].apply(converter)

data.head()
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(data['is_pop'],km.labels_))

print(classification_report(data['is_pop'],km.labels_))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train,data['is_pop'],

                                                    test_size=0.30)
km.fit(X_train,y_train)

X_test.head()
pred = km.predict(X_test)

print("Confusion Matrix \n")

print(confusion_matrix(y_test,pred))

print("\n Metrics: \n \n")

print(classification_report(y_test,pred))