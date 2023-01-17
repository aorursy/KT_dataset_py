

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



## Create DataFrame from CSV

data = pd.read_csv('../input/winemag-data_first150k.csv', index_col = 0)



## First look on the DataFrame

data.head()
## To get an overview of the data we check the most important aspects

data.describe()

##This enables a first interpretation of the data. E.g. big differences in wine prices but 75 % are cheaper than 40 $.

## Let's see how many duplicates we can find



data[data.duplicated(['description','province', 'winery','designation'])].shape[0]
## Dropping all the duplicates



data=data.drop_duplicates(['description','province', 'winery','designation'])

data=data.reset_index(drop=True)
## We will now check the amount of missing data

number_of_null_values = data.isnull().sum().sort_values(ascending = False)

percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

amount_of_null_values  = pd.concat([number_of_null_values, percent], axis=1, keys=['number_of_null_values', 'Percent'])

amount_of_null_values
## Dropping all rows with no price

data=data.dropna(subset=['price'])

data=data.reset_index(drop=True)
## Visualize the data

plt.figure(figsize= (30,10))

sns.countplot(x = 'points', data = data, palette = 'hls')

plt.show()
## Visualize average points based on the country

avg_points = data['points'].groupby(data['country']).mean()

mean_value = [avg_points[i] for i in range(len(avg_points))]



country = data.country.unique().tolist()



plt.figure(figsize=(20,5))

bars = (country)

y_values = np.arange(len(bars)-1)

plt.bar(y_values, mean_value, color = 'g')

plt.xticks(y_values, bars)

plt.xticks(rotation=86)

plt.ylim(80,95)

plt.title('Countries and the average wine score', fontsize = 20)

plt.tick_params(axis = 'both', labelsize = 15)

plt.show()

## Analysing how much percent of the wines cost more than 200 $

data[data['price']>200].shape[0]/data.shape[0]*100
## After knowing that the higher amount of wines cost less than 200$, we will check the price distribution

prices = data.price.unique().tolist()





plt.figure(figsize = (20,5))

plt.title('Distribution of price')

sns.distplot(data[data['price']< 200]['price']);

## Next we will see which countries offer on average the most expensive wines

p = data.groupby(['country'])['price','points'].mean().reset_index().sort_values('price',ascending = False)

p.head(n=10)
## Next we will see which countries offer on average the most expensive wines

p = data.groupby(['country'])['price','points'].mean().reset_index().sort_values('price',ascending = False)

p[['country','price']].head(n=10)
## Here we will check which country has the best average score

s=p.sort_values('points', ascending=False)

s[['country','points']].head(10)