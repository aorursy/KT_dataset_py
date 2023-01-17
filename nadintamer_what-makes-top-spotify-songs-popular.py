#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
#read in featuresdf.csv and store in variable named music

music = pd.read_csv("../input/featuresdf.csv")

#look at first five rows of dataset

music.head()
#look at summary of dataset

music.describe(include="all")
#we'll start off by creating two datasets, numeric and small, with some values removed

numeric = music.drop(['id','name','artists'], axis=1)

#i'm removing the below values since all the other values range between 0.0 and 1.0

#the other values are much larger, making it hard to see these values

small = numeric.drop(['tempo','duration_ms','key','loudness','time_signature'], axis=1)
#set color palette to pastel

sns.set_palette('pastel')
#create a bar chart of the mean values of the audio features in the small dataset

small.mean().plot.bar()

plt.title('Mean Values of Audio Features')

plt.show()
#mean value and distplot for danceability feature

print("Mean value for danceability:", music['danceability'].mean())

sns.distplot(music['danceability'])

plt.show()
#mean value and distplot for energy feature

print("Mean value for energy:", music['energy'].mean())

sns.distplot(music['energy'])

plt.show()
#map the numeric values of key to notes

key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}

music['key'] = music['key'].map(key_mapping)



sns.countplot(x = 'key', data=music, order=music['key'].value_counts().index)

plt.title("Count of Song Keys")

plt.show()
#mean value and distplot for loudness feature

print("Mean value for loudness:", music['loudness'].mean())

sns.distplot(music['loudness'])

plt.show()
#print mean value for mode

print("Mean value for mode feature:", music['mode'].mean())



#map the binary value of mode to major/minor

mode_mapping = {1.0: "major", 0.0: "minor"}

music['mode'] = music['mode'].map(mode_mapping)



#draw a countplot of the values

sns.countplot(x = 'mode', data=music)

plt.title("Count of Major/Minor Songs")

plt.show()
#mean value and distplot for speechiness feature

print("Mean value for speechiness:", music['speechiness'].mean())

sns.distplot(music['speechiness'])

plt.show()
#mean value and distplot for acousticness feature

print("Mean value for acousticness:", music['acousticness'].mean())

sns.distplot(music['acousticness'])

plt.show()
#mean value and distplot for instrumentalness feature

print("Mean value for instrumentalness:", music['instrumentalness'].mean())

sns.distplot(music['instrumentalness'])

plt.show()
#mean value and distplot for liveness feature

print("Mean value for liveness:valen", music['liveness'].mean())

sns.distplot(music['liveness'])

plt.show()
#mean value and distplot for valence feature

print('Mean value for valence feature:', music['valence'].mean())

sns.distplot(music['valence'])

plt.show()
#mean value and distplot for tempo feature

print('Mean value for tempo feature:', music['tempo'].mean())

sns.distplot(music['tempo'])

plt.show()
#mean value and distplot for duration_ms feature

print('Mean value for duration_ms feature:', music['duration_ms'].mean())

sns.distplot(music['duration_ms'])

plt.show()
#mean value and distplot for time_signature feature

print('Mean value for time_signature feature:', music['time_signature'].mean())

sns.distplot(music['time_signature'])

plt.show()
plt.figure(figsize = (16,5))

sns.heatmap(numeric.corr(), cmap="coolwarm", annot=True)

plt.show()