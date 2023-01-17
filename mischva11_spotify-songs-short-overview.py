import pandas as pd

import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt

import seaborn as sns
#load data and glance on them 

data = pd.read_csv("../input/top50spotify2019/top50.csv", encoding="iso-8859-1",index_col=0)



print(data.head(1),"\n\n",data.shape)

data.rename(columns={

                    "Track.Name":"track_name",

                    "Artist.Name":"artist_name",

                    "Genre":"genre",

                    "Beats.Per.Minute":"beats_per_minute",

                    "Energy":"energy",

                    "Danceability":"danceability",

                    "Loudness..dB..":"loudness",

                    "Liveness":"liveness",

                    "Valence.":"valence",

                    "Length.":"length",

                    "Acousticness..":"acousticness",

                    "Speechiness.":"speechiness",

                    "Popularity":"popularity"

                    },inplace=True)



print("\n",data.head(1))
#sort for popularity

data_popularity = data.sort_values(by=["popularity"], ascending = False)



data_popularity[0:9]

#calculate the average of the popularity and the size of each genre 

genre_popularity = data.groupby("genre").agg([np.mean, np.size])["popularity"]



#sort

genre_popularity = genre_popularity.sort_values(by=["mean"], ascending = False)





print(genre_popularity)
#calculate the average of the popularity and the size of each genre 

artist_popularity = data.groupby("artist_name").agg([np.mean, np.size])["popularity"]



#sort

artist_popularity = artist_popularity.sort_values(by=["mean"], ascending = False)



print(artist_popularity)
#Alright this shows, that a lot of musicians are in average not far distanced



#to be detailed the top 50 hast a mean of:

mu, std = norm.fit(data["popularity"])



print("\b","mean:",mu,"\n","standard deviation:",std)



#but if we check the artist the standard deviation is much smaler, the average is still

#the same



mu, std = norm.fit(artist_popularity["mean"])



print("\n","mean:",mu,"\n","standard deviation:",std)
#let's plot the popularity by artist.

value_to_plot = artist_popularity["mean"]

plt.hist(value_to_plot, density=True)

plt.plot(value_to_plot, norm.pdf(value_to_plot, mu, std))

plt.rcParams['figure.figsize'] = [35, 25]

plt.rcParams['font.size'] = 70

plt.show()

#after a short overview let's check if some of the variables are correlating the popularity, a heatmap shows a visualized overview of the correlating variables





correlations = data.corr()



ax = sns.heatmap(

    correlations, 

    vmin=-1, vmax=1, center=0,

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

)
#the popularity shows a slight positiv correlation for beats per minute and speechiness. Also a slightly negativ correlation for valence



print(correlations["popularity"][:-1])
