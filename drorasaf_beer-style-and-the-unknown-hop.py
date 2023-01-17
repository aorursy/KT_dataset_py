import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



beer_df = pd.read_csv("../input/beers.csv", index_col=0)

breweries_df = pd.read_csv("../input/breweries.csv", index_col=0)



# First take only the beer with ibu available data

# could be worth a shot to interpolate it according to the style of the beer

ibu_beer = beer_df[beer_df['ibu'].notnull()]

grouped_beer = ibu_beer.groupby('style')
total_mean = ibu_beer['ibu'].mean()

total_median = ibu_beer['ibu'].median()



print ("All beers ibu median: " + str(total_median) + " mean: "+ str(total_mean))



# mean = grouped_beer['ibu'].mean()

# median = grouped_beer['ibu'].median()

ibu_beer['ibu'].hist(bins=20)

#stout = ibu_beer['style'].str.contains('Stout', case=False)

stout = ibu_beer[ibu_beer['style'].str.contains('Stout', case=False, na=False)]

stout['ibu'].hist()
ibu_beer['abv'].hist(bins=20)
ale = ibu_beer[ibu_beer['style'].str.contains('ale', case=False, na=False)]

ale['ibu'].hist()
lager = ibu_beer[ibu_beer['style'].str.contains('lager', case=False, na=False)]

lager['ibu'].hist()