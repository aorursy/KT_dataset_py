import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/spotifyclassification/data.csv')
data.describe()
data.head()
data.info()
data.shape

red_blue = ['#19B5FE', '#EF4836']

palette = sns.color_palette(red_blue)

sns.set_palette(palette)

sns.set_style('white')
pos_tempo = data[data['target']==1]['tempo']  #selecting all the tempo values with target 1, considered as positive tempo, user liked it 

neg_tempo = data[data['target']==0]['tempo']

pos_dance =data[data['target']==1]['danceability'] 

neg_dance = data[data['target']==0]['danceability']

pos_duration_ms = data[data['target']==1]['duration_ms'] 

neg_duration_ms = data[data['target']==0]['duration_ms']

pos_loudness = data[data['target']==1]['loudness']  

neg_loudness = data[data['target']==0]['loudness']

pos_speechiness = data[data['target']==1]['speechiness']  

neg_speechiness = data[data['target']==0]['speechiness']

pos_valence = data[data['target']==1]['valence']  

neg_valence = data[data['target']==0]['valence']

pos_energy = data[data['target']==1]['energy']  

neg_energy = data[data['target']==0]['energy']

pos_acousticness = data[data['target']==1]['acousticness']  

neg_acousticness = data[data['target']==0]['acousticness']

pos_key = data[data['target']==1]['key']  

neg_key = data[data['target']==0]['key']

pos_instrumentalness = data[data['target']==1]['instrumentalness'] 

neg_instrumentalness = data[data['target']==0]['instrumentalness']



fig1 = plt.figure(figsize = (12,8))

plt.title('Song Temp')

pos_tempo.hist(alpha = 0.7, bins = 30, label='positive')

neg_tempo.hist(alpha = 0.7, bins=30,label='negative')

plt.legend (loc = 'upper right')



fig2 = plt.figure(figsize=(15,15))

#Danceability

ax3 = fig2.add_subplot(3,3,1) #positive histogram (liked)

ax3.set_xlabel('Danceability')

ax3.set_ylabel('Count')

ax3.set_title('Song Danceability Like Distribution')

pos_dance.hist(alpha=0.5, bins=30)

ax4 = fig2.add_subplot(3,3,1) #negative histogram (disliked)

neg_dance.hist(alpha = 0.5, bins=30)



#Duration_ms

ax5 = fig2.add_subplot(3,3,2)

ax5.set_xlabel('Duration_ms')

ax5.set_ylabel('Count')

ax5.set_title('Song duration Like distribution')

pos_duration_ms.hist(alpha=0.5, bins=30)

ax6 = fig2.add_subplot(3,3,2)

neg_duration_ms.hist(alpha = 0.5, bins=30)



#Loudness

ax7 = fig2.add_subplot(3,3,3)

pos_loudness.hist(alpha = 0.5, bins=30)

ax7.set_xlabel('Loudness')

ax7.set_ylabel('Count')

ax7.set_title('Song loudness like distribution')

ax8 = fig2.add_subplot(3,3,3)

neg_loudness.hist(alpha=0.5, bins=30)



#Speechiness

ax9 = fig2.add_subplot(3,3,4)

pos_speechiness.hist(alpha = 0.5, bins=30)



ax9.set_xlabel('Speechiness')

ax9.set_ylabel('Count')

ax9.set_title('Song Speechiness like distribution')

ax10 = fig2.add_subplot(3,3,4)

neg_speechiness.hist(alpha=0.5, bins=30)



#valence

ax11 = fig2.add_subplot(3,3,5)

pos_valence.hist(alpha = 0.5, bins=30)



ax11.set_xlabel('Valence')

ax11.set_ylabel('Count')

ax11.set_title('Song valence like distribution')

ax12 = fig2.add_subplot(3,3,5)

neg_valence.hist(alpha=0.5, bins=30)



#energy

ax13 = fig2.add_subplot(3,3,6)

pos_energy.hist(alpha = 0.5, bins=30)



ax13.set_xlabel('Energy')

ax13.set_ylabel('Count')

ax13.set_title('Song energy like distribution')

ax14 = fig2.add_subplot(3,3,6)

neg_energy.hist(alpha=0.5, bins=30)



#acousticness

ax15 = fig2.add_subplot(3,3,7)

pos_acousticness.hist(alpha = 0.5, bins=30)



ax15.set_xlabel('Acousticness')

ax15.set_ylabel('Count')

ax15.set_title('Song acousticness like distribution')

ax15 = fig2.add_subplot(3,3,7)

neg_acousticness.hist(alpha=0.5, bins=30)



#key

ax16 = fig2.add_subplot(3,3,8)

pos_key.hist(alpha = 0.5, bins=30)



ax16.set_xlabel('Key')

ax16.set_ylabel('Count')

ax16.set_title('Song key like distribution')

ax16 = fig2.add_subplot(3,3,8)

neg_key.hist(alpha=0.5, bins=30)



#instrumentalness

ax17 = fig2.add_subplot(3,3,9)

pos_instrumentalness.hist(alpha = 0.5, bins=30)



ax17.set_xlabel('Instrumentalness')

ax17.set_ylabel('Count')

ax17.set_title('Song instrumentalness like distribution')

ax17 = fig2.add_subplot(3,3,9)

neg_instrumentalness.hist(alpha=0.5, bins=30)