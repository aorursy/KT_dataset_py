import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

print(os.listdir("../input"))



import warnings  

warnings.filterwarnings('ignore')
df = pd.read_excel('../input/farhans-spotify-top-100-2018/Top100.xlsx')

df2 = pd.read_excel('../input/spotifys-top-100-most-played/Top100(spotify).xlsx')
sns.set(style='whitegrid')
#Valence - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive(e.g. happy, cheerful, euphoric), #while tracks with low valence sound more negative (e.g. sad, depressed, angry)



#The valence distribution for my top 100 songs of 2018 seems normally distributed, displaying no preference for overly cheery or depressing songs.



#Spotify's top 100 most played displays a higher average valence



sns.distplot(df['valence'],bins=15, kde=True, rug=True, color='orange')

sns.distplot(df2['valence'],bins=15, kde=True, rug=True, color='grey')
df.mean()
df2.mean()
#The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.

#The loudness distribution for my top 100 songs of 2018 is skewed to the right, indicating a slight preference for louder music



#Spotify's top 100 most played tracks are louder than my top 100. A speculative reason could be the Loudness War, referring to the trend of increasing audio levels in recorded music which many critics believe reduces sound quality and dynamic range of the track, apparent in pop songs where ultra-compression occurs.



sns.distplot(df['loudness'],bins=15, kde=True, rug=True, color='red')

sns.distplot(df2['loudness'],bins=15, kde=True, rug=True, color='grey')
#Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

#Normally distributed, indicating little preference. I listen to a variety of music so this make sense as I'm not restricted to a particular music style.

#Spotify's top 100 most played displays higher energy



sns.distplot(df['energy'],bins=15, kde=True, rug=True, color='green')

sns.distplot(df2['energy'],bins=15, kde=True, rug=True, color='grey')
#A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

#Preference is shown for non-acoustic songs. I listen to acoustic tracks but generally prefer a fuller band setting with other instruments.



sns.distplot(df['acousticness'],bins=15, kde=True, rug=True, color='purple')

sns.distplot(df2['acousticness'],bins=15, kde=True, rug=True, color='grey')
#Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

#The distribution histogram shows a preference towards less spoken word in proportion to instrumentation. Although I listen to a great deal of instrumental music, I also listen to Rap & R&B which should indicate a higher speechiness score.



sns.distplot(df['speechiness'],bins=15, kde=True, rug=True, color='pink')

sns.distplot(df2['speechiness'],bins=15, kde=True, rug=True, color='grey')
#The duration of the track in milliseconds.

#Most of the songs seem to last in the range of about 200,000ms = 3.3 mins~. There are outliers due to the nature of the genre. Most songs I listen to tend to be around this range with the exception of prog-rock and instrumentals, which tend to be longer and may represent the outliers that we see below.

#Both datasets display similar duration for the songs. Spotify's top 100 reveals that most mainstream songs generally have a similar play time, perhaps radio-friendly lengths.

sns.distplot(df['duration_ms'], bins=10, color='teal')

sns.distplot(df2['duration_ms'], bins=10, color='grey')
#Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.

#The graph displays a slight skew to the right, indicating my preference for more 'danceable songs'. The moderately high number of R&B tracks(typically with a strong beat) could have contributed to this.



sns.distplot(df['danceability'], bins=25, color='blue')

sns.distplot(df2['danceability'], bins=25, color='grey')
#The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.



sns.distplot(df['tempo'], bins=25, color='magenta')

sns.distplot(df2['tempo'], bins=25, color='grey')
#exploratory data analysis for my top 100 songs of 2018



sns.pairplot(df)
#exploratory data analysis for spotify's top 100 most played



sns.pairplot(df2)
#We can observe from the pairplots above that there is a correlation between the loudness and the perceived energy of the song. The jointplots below are from both datasets and indicate a correlation between the two.

#The louder the song is, the more energetic it is perceived to be. There is a correlation between the "danceability" and "valence" of a track but it is not as obvious or apparent as the previous correlation.



sns.jointplot('loudness','energy', data=df2,kind='reg')

sns.jointplot('loudness','energy', data=df, kind='reg')