# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib as mlp

import matplotlib.pyplot as plt

import seaborn as sns

import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
wine_data = ("/kaggle/input/wine-reviews/winemag-data_first150k.csv")

wine_data1 = ("/kaggle/input/wine-reviews/winemag-data-130k-v2.csv")



reviews = pd.read_csv(wine_data, index_col = 'country', sep =",")

reviews1 = pd.read_csv(wine_data1, index_col = 'country', sep = ",")



reviews = pd.DataFrame(reviews)

reviews1 = pd.DataFrame(reviews1) 



reviews = reviews.append(reviews1,ignore_index = False)
reviews.head()
reviews1.head()
reviews = reviews[reviews['price'].notna()]

plt.figure(figsize=(40,30))

plt.xticks(rotation=90)

plt.rcParams.update({'font.size':30})

plt.title("Country average price of wine")

'''

average_england=reviews.loc[reviews.index=='England']

average_england=average_england['price'].mean()

print (average_england)

'''







sns.barplot(reviews.index, reviews.price)



plt.show()
wines_1500 = reviews.loc[reviews['price'] < 1500]

wines_1500
plt.figure(figsize=(35,20))

plt.xticks(rotation=90)

plt.title("Cost of wine to points scored")



sns.scatterplot(wines_1500['price'], wines_1500['points'])



plt.show()
reviews_90 = reviews.loc[reviews['points'] > 98]

reviews_90
plt.figure(figsize=(10,20))

plt.xticks(rotation=90)

plt.xlim([97, 100])

plt.rcParams.update({'font.size':14})

plt.title("Points earned by winery")

sns.barplot(reviews_90['points'],reviews_90['winery'],data=reviews_90)



reviewsprice_1000 = reviews.loc[reviews['price'] > 1000]

reviewsprice_1000
plt.figure(figsize=(10,10))

plt.xticks(rotation=90)

sns.barplot(reviewsprice_1000['variety'], reviewsprice_1000['price'])

reviewspoints_99 = reviews.loc[reviews['points'] > 98]

reviewspoints_99
plt.figure(figsize=(5,30))

plt.xticks(rotation=90)

plt.rcParams.update({'font.size':40})

plt.title("Top Winerie's Around The World")



sns.barplot(reviewspoints_99['points'], reviewspoints_99['winery'])

#print (reviewspoints_99["winery"])



reviewspoints_95 = reviews.loc[reviews['points'] > 95]

reviewspoints_95
plt.figure(figsize=(5,30))

plt.xticks(rotation=90)

plt.xlim([94, 100])



plt.rcParams.update({'font.size':40})

plt.title("Top Province's Around The World")



sns.barplot(reviewspoints_95['points'], reviewspoints_95['province'])

#print (reviewspoints_99["winery"])
best_wine = reviews['variety'].value_counts()[:2]

wine_top_3 = reviews[reviews['variety'].isin(best_wine.index)]

wine_top_3.shape
wine_top_3 = wine_top_3.dropna(axis=1)

wine_top_3
plt.figure(figsize=(15,5))



plt.title("top 3 wines from variety")

#MOST reviewed wine by vairity

reviews_1000 = reviews.loc[reviews['price'] > 1000]

reviews_1000
plt.figure(figsize=(10,5))

plt.title("most expensive wine in countys")



sns.barplot(reviewsprice_1000.index, reviewsprice_1000['price'])



top_us_wine = reviews.loc[(reviews.index == 'US') & (reviews.points > 99)]

top_us_wine
plt.figure(figsize=(10,10))

plt.xticks(rotation=0)



sns.barplot(top_us_wine['price'],top_us_wine['variety'])



#you want a highl rated wine from the US, these are the ones you choose
top_france_wine = reviews.loc[(reviews.index == 'France') & (reviews.points > 95)]

top_france_wine
plt.figure(figsize=(10,17))

plt.xticks(rotation=0)

plt.xlim([90, 100])

plt.title("French blends with a min avg of 96")



sns.barplot(top_france_wine['points'],top_france_wine['variety'])