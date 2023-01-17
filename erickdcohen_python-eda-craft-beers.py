import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

beerRaw = pd.read_csv("../input/craft-cans/beers.csv")

breweriesRaw = pd.read_csv("../input/craft-cans/beers.csv")
print(list(beerRaw.columns.values))



print(list(breweriesRaw.columns.values))



# Adding index to breweriesRaw

breweriesRaw['brewery_id'] = breweriesRaw.index





beerDf = pd.merge(beerRaw, breweriesRaw, how = "outer")



print(beerDf.head)



print(list(breweriesRaw.columns.values))
# What columns are in our dataset? 



print(beerDf.columns.values)
# Digging a little deeper

print(beerDf.ibu.describe())
# Describe with grouping

beerDf.groupby(by = "style").describe()
# Counts of the breweries in the dataset 

beerDf['name'].value_counts()

# What are the most common styles?



beerDf['style'].value_counts()[:15]
hefeDf = beerDf[beerDf['style'] == "Hefeweizen"]



hefeDf
# Average ABV of a hefeweizen

hefeDf.abv.mean()
# Median ABV of a hefeweizen

hefeDf.abv.median()
# Average IBU of a hefeweizen 

hefeDf.ibu.mean()
# Median IBU of a hefeweizen

hefeDf.ibu.median()
hefeDf.describe() # Same info with describe()
beerDf['name'].value_counts()[:20].plot(kind = "bar")



plt.title("Count of Brewers")



plt.show()

plt.scatter("ibu", "abv", data = beerDf)



plt.title("IBU vs. ABV")
beerDf['style'].value_counts()[:15].plot(kind = "bar", color = "red")



plt.title("Most Common Beer Styles")
# Let's also see how the abv is distributed



plt.hist("abv", 8, data = beerDf, alpha = 0.8)



plt.title("Distribution of ABV")
plt.hist("ibu", 11, data = beerDf, alpha = 0.8, color = "green")



plt.title("Distribution of IBU")