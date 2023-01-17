import pandas as pd

import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt

import seaborn as sns
#load data and glance on them 

data = pd.read_csv("../input/top-50-spotify-songs-by-each-country/top50contry.csv", encoding="iso-8859-1",index_col=0)



print(data.head(1),"\n\n",data.shape)
data_germany = data[data.country == "germany"]



print("\n",data_germany.head(1))
#sort for popularity

data_popularity = data_germany.sort_values(by=["pop"], ascending = False)



data_popularity[0:9]
#calculate the average of the popularity and the size of each genre 

genre_popularity = data_germany.groupby("top genre").agg([np.mean, np.size])["pop"]



#sort

genre_popularity = genre_popularity.sort_values(by=["mean"], ascending = False)





print(genre_popularity)
#calculate the average of the popularity and the size of each genre 

artist_popularity = data_germany.groupby("artist").agg([np.mean, np.size])["pop"]



#sort

artist_popularity = artist_popularity.sort_values(by=["mean"], ascending = False)



print(artist_popularity)

#Alright this shows



#to be detailed the top 50 hast a mean of:

mu, std = norm.fit(data["pop"])



print("\b","mean:",mu,"\n","standard deviation:",std)



#if we take the average of the popularity the average musician we are nearly close.

mu, std = norm.fit(artist_popularity["mean"])



print("\n","mean:",mu,"\n","standard deviation:",std)
#let's plot the popularity by artist.

value_to_plot = artist_popularity["mean"]

plt.hist(value_to_plot, density=True)

plt.plot(value_to_plot, norm.pdf(value_to_plot, mu, std))

plt.title('german musicians popularity')

plt.show()







correlations = data_germany.corr()



ax = sns.heatmap(

    correlations, 

    vmin=-1, vmax=1, center=0,

    square=True,

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

)

ax.set_title("correlation matrix of the variables (germany)")
print(correlations["pop"][:-1])
#calculate the average of the popularity and the size of each genre 

genre_popularity_worlwide = data.groupby("top genre").agg([np.mean, np.size])["pop"]



#sort

genre_popularity_worlwide = genre_popularity_worlwide.sort_values(by=["mean"], ascending = False)



print(genre_popularity_worlwide.head(30))
data[data['top genre']=='australian pop'].groupby("title").size()
print(data.corr()["pop"][:-1])
#let's plot the popularity distribution by country.

country_data = data.groupby("country").agg([np.mean])["pop"]



value_to_plot = country_data["mean"]

mu, std =  norm.fit(value_to_plot)

print ("average worlwide: ", mu, " standard deviation worldwide: ", std)



plt.hist(value_to_plot, density=True)

plt.plot(value_to_plot, norm.pdf(value_to_plot, mu, std))

plt.title('worlwide musicians popularity')

plt.show()