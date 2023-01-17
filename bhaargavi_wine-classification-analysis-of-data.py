import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline
import nltk
# Importing Natural language Processing toolkit.
from PIL import Image
# from python imaging library
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wines = pd.read_csv("../input/winemag-data-130k-v2.csv",index_col = 0)
wines = wines.dropna(subset = ['points' , 'price'])
print(wines.shape)
wines.head(2)
print("There are {} countries and these are {}\n".format(len(wines.country.unique()), ", ".join(wines.country.unique()[0:5])))
print("There are {} varities of wine which are {}\n".format(len(wines.variety.unique()), ", ".join(wines.variety.unique()[0:5])))
wines_country = pd.crosstab(index = wines['country'], columns = 'count')
plt.rcParams["figure.figsize"][0] = 10
plt.rcParams["figure.figsize"][1] = 4
wines_country['count'].plot(kind = 'bar',width= 0.7)
plt.title("Graph showing total number of entries of each country", fontsize = 17)
plt.xlabel("Countries", fontsize = 14)
plt.ylabel("Frequency of entries", fontsize = 14)
variety = pd.crosstab(index = wines['variety'], columns = 'count').sort_values('count' , ascending = False)
wines_var = variety[variety['count'] > 1000]
wines_var['count'].plot(kind = 'bar', width = 0.7)
plt.title("The number of each variety of wine present", fontsize= 17)
plt.xlabel("The variety of wines that are present", fontsize = 14)
plt.ylabel("The number of wine variety that are present", fontsize = 14)
wines['points'].hist(grid = False, bins = 40,color = 'purple')
wines['points'].describe()
price = pd.crosstab(index = wines['price'], columns = 'count').sort_values(by = 'count', ascending = False)
plt.rcParams['figure.figsize'][0] = 4
plt.rcParams['figure.figsize'][1] = 4
price['count'].hist(grid = False)
price_max = wines[wines['price'] < 100]['price']
price_max.hist(grid = False, color = 'green')
price_max.describe()
