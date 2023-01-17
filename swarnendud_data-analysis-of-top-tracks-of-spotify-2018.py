import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/top-spotify-tracks-of-2018/top2018.csv")
df.info()
df.describe()
df.head()
## Initialize the figure

plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(20,8))

df["danceability"].plot(color="blue")

plt.xlabel('Top 100 Tracks',fontsize=20)

plt.ylabel('danceability',fontsize=20)

plt.title('danceability Plot',fontsize=20)
#Let's plot all features for top 100 tracks

df_features_list = ['energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',

       'valence', 'tempo', 'duration_ms', 'time_signature']



for i in df_features_list:

    ## Initialize the figure

    plt.style.use('seaborn-darkgrid')

    plt.figure(figsize=(20,8))

    df[i].plot(color="blue")

    plt.xlabel('Top 100 Tracks',fontsize=20)

    plt.ylabel(i,fontsize=20)

    plt.title(i +' Plot',fontsize=20)

    

sns.heatmap(df.corr())
df['artists'].value_counts().head(10)