#import appropriate libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#read dataframe

wine_df=pd.read_csv("../input/winemag-data_first150k.csv")
#explore dataframe

wine_df.head()
#basic visualization of values

plt.xlabel('Point Values')

plt.ylabel('Count')

plt.title('Overview of Wine DF')

plt.hist(wine_df['points'],bins = 15, edgecolor = 'white')
#explore missing values

sns.heatmap(wine_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#drop columns that are missing data

wine_df.drop('region_1',axis=1,inplace=True)
wine_df.drop('region_2',axis=1,inplace=True)
wine_df.drop('description',axis=1,inplace=True)
wine_df.drop('designation',axis=1,inplace=True)
#check out new dataframe structure

wine_df.head()
#create new columns to identify which are the best rated and less expensive wines

wine_df["cheaper"] = 'no'

wine_df["cheaper"][wine_df["price"]< 20.0] = 'yes'

wine_df["quality"] = 'no'

wine_df["quality"][wine_df["points"]> 91] = 'yes'
#create new dataframe with only quality, inexpensive wines

newdf= wine_df[(wine_df['cheaper']=="yes") & (wine_df['quality']=="yes")]
newdf.head()
#this graph shows that the US produces the most inexpensive higher quality wines 

#(if white wines are included)

#sns.countplot(x='country',data=newdf,)

ax = sns.countplot(x="country", data=newdf)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()
#this graph shows that Sauv Blanc is the varietal that has the most value for your $

plt.figure(figsize=(10,6))

newdf['variety'].value_counts().plot(kind='bar', title = "Countries with highest rated low-cost wines(all varietals)")
#create new column that identifies which are red wines

#can someone do this with .loc to kill the error reports and still retain the all cases designation?



wine_df["reds"] = 'no'

wine_df["reds"][wine_df["variety"].str.contains("Red",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Cabernet",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Pinot Noir",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Syrah",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Malbec",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Sangiovese",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Merlot",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Grenache",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Shiraz",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Pinotage",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Monastrell",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Tempranillo",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Claret",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Mourvèdre",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Verdot",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Dolcetto",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("Carmenère",case=False)]= 'yes'

wine_df["reds"][wine_df["variety"].str.contains("G-S-M",case=False)]= 'yes'

#creates a new data frame that lists only red quality, inexpensive wines

red_df = wine_df[(wine_df['cheaper']=="yes") & (wine_df['quality']=="yes") & (wine_df['reds']=="yes")]
red_df.head()
#this graph shows that the best inexpensive red wines come primarily from Portugal. 

#sns.countplot(x='variety',data=red_df)

red_df['country'].value_counts().plot(kind='bar', title='Countries with the highest rated, low cost red wines')
#this plot shows the varieties of reds that tend to be the best yet inexpensive

red_df['variety'].value_counts().plot(kind='bar', title='Number of Red Varietals 92+ Rating')
#here is the complete dataframe that lists the best red wines that are under $20

#red_df.drop(['designation'],axis=1, inplace=True)
red_df