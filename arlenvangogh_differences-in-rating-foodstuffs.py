# importing the data

import numpy as np

import pandas as pd

ramen = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv")

wine = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

chocolate = pd.read_csv("../input/chocolate-bar-ratings/flavors_of_cacao.csv")
# importing the visualization tools and initializing them

import seaborn as sns

import matplotlib.pyplot as plt
# Creating a histogram of the ramen data



# Setting up 

RamenData = pd.DataFrame(np.round(ramen['Stars'].replace('Unrated', np.nan).dropna().astype(np.float64)))

RamenData = RamenData.groupby(by = 'Stars').size()



# Plotting

pal = sns.color_palette('BuGn_r', len(RamenData))

rank = RamenData.argsort().argsort()

sns.barplot(x=RamenData.index, y=RamenData, palette=np.array(pal[::-1])[rank])

plt.suptitle("Ramen Ratings")                 
# Creating a histogram of the wine data



# Setting up

WineData = pd.DataFrame(np.round((wine['points'].dropna() - 80) / 4))

WineData = WineData.groupby(by = 'points').size()

rank = WineData.argsort().argsort()



# Plotting

pal = sns.color_palette('BuGn_r', len(WineData))

sns.barplot(x=WineData.index, y=WineData, palette=np.array(pal[::-1])[rank])

plt.suptitle("Wine Ratings")    
# Creating a histogram of the chocolate data



#Setting up 

ChocolateData = pd.DataFrame(np.round(chocolate['Rating'].dropna()))

ChocolateData = ChocolateData.groupby(by = 'Rating').size()

rank = ChocolateData.argsort().argsort()



# Plotting

pal = sns.color_palette('BuGn_r', len(ChocolateData))

sns.barplot(x=ChocolateData.index, y=ChocolateData, palette=np.array(pal[::-1])[rank])

plt.suptitle("Chocolate Ratings")   