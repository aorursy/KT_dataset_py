#import libraries
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scikitplot as plot
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
import math 
%matplotlib inline
#combine datasets and remove twitter handles
wine_filepath1 = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
wine_filepath2 = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
#wine_filepath3 = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.json", index_col=0)

winedf = pd.concat([wine_filepath1, wine_filepath2])
#.drop(['taster_twitter_handle'], axis=1)
#show data
print(winedf)

#Number of unique wine columns
winedf.nunique()
#Find out countries 
#print(winedf["country"].unique())

#Seperate US and other Countries
USA = winedf.loc[winedf["country"] == "US"]
print(USA)


IntCountries = winedf.loc[winedf["country"] != "US"]
print(IntCountries)
#Combine Points and variety to get Hi rated wines

#hiratedwines = winedf[['points', 'variety']].head(20)
#hiratedwines 

top10wine= winedf("variety")["Cabernet Sauvignon", "Tinta de Toro", "Sauvignon Blanc", "Pinot Noir", "Provence red blend", "Friulano", "Chardonnay", "Cabernet Sauvignon", "Tempranillo", "Malbec"]
top10wine

sns.catplot(x= "country", y= "points", data=winedf ,errwidth = 2, capsize = .05, kind="bar")
plt.xticks(rotation=90)
plt.title("Most Rated Wines by Countries")
plt.figure(figsize=(30,20))

plt.show ()



USA = winedf.loc[winedf["country"] == "US"]
sns.catplot(x= USA , y= "points", data=winedf, errwidth = 2, capsize = 0.05, kind = "bar")
plt.title("Hi Rated Wine in US")
plt.figure(figsize=(12,10))

plt.show()     

#sns.barplot(x= USA , y= hiratedwines, data=winedf, errwidth = 2, capsize = 0.05)
#plt.show()

IntCountries = winedf.loc[winedf["country"] != "US"]
sns.catplot(x= "INTCountries" , y= "points", data=winedf, errwidth = 2, capsize = 0.05, kind = "bar")
plt.title("Hi Rated Wine in US")
plt.figure(figsize=(12,10))

plt.show()        

#sns.barplot(x= INTCountries , y= hiratedwines, data=winedf, errwidth = 2, capsize = 0.05)
#plt.show()

#winedf.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

#Get Top 10 amounts of reviews by their country and province * len counts characters not entries

#countries_reviewed = winedf.groupby(['country', 'province']).description.agg([len]) #length of descr
#countries_reviewed = winedf.groupby(['country', 'province']).description.agg([max]) #the most length max to min
#countries_reviewed

#countries_reviewed.sort_values(by='len', ascending=False).head(10)

