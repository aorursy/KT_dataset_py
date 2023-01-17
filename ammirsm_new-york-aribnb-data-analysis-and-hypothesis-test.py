# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd 

import numpy as np

import seaborn as sns 

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import geopandas

from scipy import stats

from statsmodels.stats import weightstats as mstats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = AB_NYC_2019 = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.head()
host_names = data['host_name'].dropna()

reviews_per_month = data['reviews_per_month'].dropna()

names = data['name'].dropna()

neighbourhood = data['neighbourhood']
# Global function to make wordcloud infographics

def make_wordcloud(words):



    text = ""

    for word in words:

        text = text + " " + word



    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(stopwords=stopwords,colormap="plasma",width=1920, height=1080,max_font_size=200, max_words=200, background_color="white").generate(text)

    plt.figure(figsize=(20,20))

    plt.imshow(wordcloud, interpolation="gaussian")

    plt.axis("off")

    plt.show()

make_wordcloud(host_names)
make_wordcloud(neighbourhood)


plt.figure(figsize=(10,10))

plt.title("Number of houses in each neighbourhood group")

ax = sns.countplot(data["neighbourhood_group"], palette="bright" ) 
nyc = geopandas.read_file(geopandas.datasets.get_path('nybb'))

nyc = nyc.to_crs(epsg=4326)

crs = {'init':'epsg:4326'}

geometry = geopandas.points_from_xy(data.longitude, data.latitude)

geo_data = geopandas.GeoDataFrame(data,crs=crs,geometry=geometry)



fig,ax = plt.subplots(figsize=(15,15))

nyc.plot(ax=ax,alpha=0.4,edgecolor='black')

geo_data.plot(column='room_type',ax=ax,legend=True,cmap='plasma',markersize=4)



plt.title("Plot to demonstrate the distribution of the location of house types")

plt.axis('off')


fig,ax = plt.subplots(figsize=(15,15))

nyc.plot(ax=ax,alpha=0.4,edgecolor='black')

geo_data.plot(column='id',ax=ax,legend=True,cmap='plasma',markersize=4)



plt.title("Distribution of number of listings in each area")

plt.axis('on')


plt.figure(figsize=(10,10))

plt.title("Number of different types of homes in different neighborhood groups")

ax = sns.countplot(data['room_type'],hue=data['neighbourhood_group'], palette='pastel')
#Number of available days

fig,ax = plt.subplots(figsize=(15,15))

nyc.plot(ax=ax,alpha=0.4,edgecolor='black')

geo_data.plot(column='availability_365',ax=ax,legend=True,cmap='plasma',markersize=4)



plt.title("Number of available days for booking")

plt.axis('off')
plt.figure(figsize=(10,10))

ax = sns.boxplot(data=data, x='neighbourhood_group',y='availability_365',palette='colorblind')

plt.show()
data['price'].groupby(data["neighbourhood_group"]).describe().round(2)

#By looking at the mean values we can understand that Manhattan is the most and Bronx is the least expensive neighborhood

plt.figure(figsize=(12,12))

ax = sns.heatmap(data.corr(),annot=True)

plt.title("Dataset Heatmap")
data.corr().style.background_gradient(cmap='seismic')
data['neighbourhood_group'].value_counts()

# Manhattan and Brooklyn have same number of samples.
y = data["price"]

plt.figure(1); plt.title('Normal')

sns.distplot(y, kde=False, fit=stats.norm)



plt.figure(2); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=stats.johnsonsu)



fig = plt.figure()

res = stats.probplot(y, plot=plt)

plt.show()
from scipy import stats

def is_normal(x, alpha = 1e-3):

    k2,p = stats.normaltest(x)

    print('--p-- = {}'.format(p))

    return p > alpha





#Shuffle it!

data = data.sample(frac=1)



manhattan_sample = data[data['neighbourhood_group'] == 'Manhattan']

manhattan_sample = manhattan_sample[:50]



#if the qq plot points are on the line, the data is normally distributed

fig = plt.figure()

res = stats.probplot(manhattan_sample['price'], plot=plt)

plt.show()

        

print(is_normal(np.array(manhattan_sample['price'])))



brooklyn_sample = data[data['neighbourhood_group'] == 'Brooklyn']

brooklyn_sample = brooklyn_sample[:50]



#if the qq plot points are on the line, the data is normally distributed

fig = plt.figure()

res = stats.probplot(brooklyn_sample['price'], plot=plt)

plt.show()



print(is_normal(np.array(brooklyn_sample['price'])))



print(stats.zscore(manhattan_sample['price']))

print(stats.zscore(brooklyn_sample['price']))
#in reality you might estimate this value another way for a very large population

population_mean = data['price'].mean()

#sample mean 'math_score' (just for illustrative purposes, not strictly required)

manhattan_sample_mean = manhattan_sample['price'].mean()

brooklyn_sample_mean = brooklyn_sample['price'].mean()



print(population_mean,manhattan_sample_mean)

print(population_mean,brooklyn_sample_mean)





#this is a 'one sample' test

zstat, pvalue = mstats.ztest(manhattan_sample['price'],x2=None,value=population_mean,alternative='smaller')



#p-value is not small; this is enough evidence to say it is smaller than the whole population

print(float(pvalue))





#this is a 'one sample' test

zstat, pvalue = mstats.ztest(brooklyn_sample['price'],x2=None,value=population_mean,alternative='smaller')



#p-value is very very small < 0.1%; this is enough evidence to reject the null hypothesis

#of course this was expected and obvious, but this is supposed to be a transparent,clear example

print(float(pvalue))
