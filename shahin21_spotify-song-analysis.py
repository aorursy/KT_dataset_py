import pandas as pd

import numpy as np

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import seaborn as sns



#%matplotlib inline
spotifyData = pd.read_csv("../input/spotifyclassification/data.csv")
type(spotifyData)
spotifyData.shape
spotifyData.info()
spotifyData.head()
spotifyData.describe()
spotifyFloatData = spotifyData.select_dtypes(include=['float64'])

spotifyFloatData.head(10)
spotifyIntegerData = spotifyData.select_dtypes(include=['int64'])

spotifyIntegerData.head(10)
# custom color

green_red = ['green', 'red']

palette = sns.color_palette(green_red)

sns.set_palette(palette)

sns.set_style('white')
positiveTempo = spotifyData[spotifyData['mode'] == 1]['tempo']

negativeTempo = spotifyData[spotifyData['mode'] == 0]['tempo']

plt.figure(figsize=(50,10))

positiveTempo.plot()

negativeTempo.plot()
plt.figure(figsize=(16,8))

plt.title('Song tempo mode distribution')

plt.xlabel('Tempo')

plt.ylabel('Song Count')

positiveTempo.hist(bins=100, alpha=.5, label='Positive')

negativeTempo.hist(bins=100, alpha=.5, label='Negative')

plt.legend()
posAcousticness = spotifyData[spotifyData['mode'] == 1]['acousticness']

negAcousticness = spotifyData[spotifyData['mode'] == 0]['acousticness']

posDanceability = spotifyData[spotifyData['mode'] == 1]['danceability']

negDanceability = spotifyData[spotifyData['mode'] == 0]['danceability']

posDuration = spotifyData[spotifyData['mode'] == 1]['duration_ms']

negDuration = spotifyData[spotifyData['mode'] == 0]['duration_ms']

posEnergy = spotifyData[spotifyData['mode'] == 1]['energy']

negEnergy = spotifyData[spotifyData['mode'] == 0]['energy']

posInstru = spotifyData[spotifyData['mode'] == 1]['instrumentalness']

negInstru = spotifyData[spotifyData['mode'] == 0]['instrumentalness']

posKey = spotifyData[spotifyData['mode'] == 1]['key']

negKey = spotifyData[spotifyData['mode'] == 0]['key']

posLiveness = spotifyData[spotifyData['mode'] == 1]['liveness']

negLiveness = spotifyData[spotifyData['mode'] == 0]['liveness']

posLoudness = spotifyData[spotifyData['mode'] == 1]['loudness']

negLoudness = spotifyData[spotifyData['mode'] == 0]['loudness']

posTime = spotifyData[spotifyData['mode'] == 1]['time_signature']

negTime = spotifyData[spotifyData['mode'] == 0]['time_signature']

posValence = spotifyData[spotifyData['mode'] == 1]['valence']

negValence = spotifyData[spotifyData['mode'] == 0]['valence']

posTarget = spotifyData[spotifyData['mode'] == 1]['target']

negTarget = spotifyData[spotifyData['mode'] == 0]['target']
figSize = plt.figure(figsize=(30,20))



# acousticness

ax1 = figSize.add_subplot(331)

ax1.set_title('Song acousticness - mode distribution')

ax1.set_xlabel('Acousticness')

ax1.set_ylabel('Song Count')

posAcousticness.hist(bins=30, alpha=.5, label='P')

negAcousticness.hist(bins=30, alpha=.5, label='N')

ax1.legend()



# danceability

ax2 = figSize.add_subplot(332)

ax2.set_title('Song danceability - mode distribution')

ax2.set_xlabel('Danceability')

ax2.set_ylabel('Song Count')

posDanceability.hist(bins=30, alpha=.5, label='P')

negDanceability.hist(bins=30, alpha=.5, label='N')

ax2.legend()



# duration_ms

ax3 = figSize.add_subplot(333)

ax3.set_title('Song duration_ms - mode distribution')

ax3.set_xlabel('duration_ms')

ax3.set_ylabel('Song Count')

posDuration.hist(bins=30, alpha=.5, label='P')

negDuration.hist(bins=30, alpha=.5, label='N')

ax3.legend()



# energy

ax4 = figSize.add_subplot(334)

ax4.set_title('Song energy - mode distribution')

ax4.set_xlabel('energy')

ax4.set_ylabel('Song Count')

posEnergy.hist(bins=30, alpha=.5, label='P')

negEnergy.hist(bins=30, alpha=.5, label='N')

ax4.legend()



# instrumentalness

ax5 = figSize.add_subplot(335)

ax5.set_title('Song instrumentalness - mode distribution')

ax5.set_xlabel('instrumentalness')

ax5.set_ylabel('Song Count')

posInstru.hist(bins=30, alpha=.5, label='P')

negInstru.hist(bins=30, alpha=.5, label='N')

ax5.legend()



# key

ax6 = figSize.add_subplot(336)

ax6.set_title('Song key - mode distribution')

ax6.set_xlabel('key')

ax6.set_ylabel('Song Count')

posKey.hist(bins=30, alpha=.5, label='P')

negKey.hist(bins=30, alpha=.5, label='N')

ax6.legend()



# liveness

ax7 = figSize.add_subplot(337)

ax7.set_title('Song liveness - mode distribution')

ax7.set_xlabel('liveness')

ax7.set_ylabel('Song Count')

posLiveness.hist(bins=30, alpha=.5, label='P')

negLiveness.hist(bins=30, alpha=.5, label='N')

ax7.legend()



# time_signature

ax8 = figSize.add_subplot(338)

ax8.set_title('Song time_signature - mode distribution')

ax8.set_xlabel('time_signature')

ax8.set_ylabel('Song Count')

posTime.hist(bins=30, alpha=.5, label='P')

negTime.hist(bins=30, alpha=.5, label='N')

ax8.legend()



# valence

ax9 = figSize.add_subplot(339)

ax9.set_title('Song valence - mode distribution')

ax9.set_xlabel('valence')

ax9.set_ylabel('Song Count')

posValence.hist(bins=30, alpha=.5, label='P')

negValence.hist(bins=30, alpha=.5, label='N')

ax9.legend()
# target

plt.title('Song target - mode distribution')

plt.xlabel('target')

plt.ylabel('Song Count')

posTarget.hist(bins=10, alpha=.5, label='Positive')

negTarget.hist(bins=10, alpha=.5, label='Negative')

plt.legend()
train, test = train_test_split(spotifyData, test_size = 0.15)
print(f"Traing size = {len(train)}, Test size = {len(test)}")
train.shape