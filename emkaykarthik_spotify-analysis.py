# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df  = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding='latin1')
df.head()
df.columns.to_list()
### total tracks
len(df)
for val in df.columns:
    print(val)
    print(df[val].unique())
    print("*"*50)
df_bpm = df.sort_values(by=['Beats.Per.Minute'])
import matplotlib.pyplot as plt
import pandas as pd

dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Beats.Per.Minute" : df['Beats.Per.Minute'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Beats.Per.Minute", "Popularity"],figsize=(20,5), grid=True)
plt.show()
df_energy = df.sort_values(by=['Energy'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Energy" : df['Energy'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Energy", "Popularity"],figsize=(20,5), grid=True)
plt.show()
df_energy = df.sort_values(by=['Danceability'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Danceability" : df['Danceability'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Danceability", "Popularity"],figsize=(20,5), grid=True)
plt.show()
df.columns
df_energy = df.sort_values(by=['Loudness..dB..'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Loudness..dB.." : df['Loudness..dB..'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Loudness..dB..", "Popularity"],figsize=(20,5), grid=True)
plt.show()
df_energy = df.sort_values(by=['Liveness'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Liveness" : df['Liveness'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Liveness", "Popularity"],figsize=(20,5), grid=True)
plt.show()
df.columns
df_energy = df.sort_values(by=['Valence.'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Valence." : df['Valence.'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Valence.", "Popularity"],figsize=(20,5), grid=True)
plt.show()
df_energy = df.sort_values(by=['Length.'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Length." : df['Length.'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Length.", "Popularity"],figsize=(20,5), grid=True)
plt.show()
df_energy = df.sort_values(by=['Acousticness..'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Acousticness.." : df['Acousticness..'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Acousticness..", "Popularity"],figsize=(20,5), grid=True)
plt.show()

df_energy = df.sort_values(by=['Speechiness.'])
dff = pd.DataFrame({"Track.Name": df['Track.Name'],
                   "Speechiness." : df['Speechiness.'],
                   "Popularity" : df['Popularity'] })
dff.plot(x="Track.Name", y=["Speechiness.", "Popularity"],figsize=(20,5), grid=True)
plt.show()

df
## Songs which are featured by the Other artists
df['Track.Name'] = df['Track.Name'].str.lower()
featured_songs = df[df['Track.Name'].str.contains("feat")]
featured_songs
single_artist_songs = df[df["Track.Name"].str.contains('feat') == False]

single_artist_songs.head()
# Popularity of the individual artist songs and featured songs
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
axes[0].scatter(featured_songs.index, featured_songs['Popularity'])
axes[1].scatter(single_artist_songs.index, single_artist_songs['Popularity'])
fig.tight_layout()