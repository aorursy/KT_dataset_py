#Importing Libraries

import os # accessing directory structure

import numpy as np

import requests

import pandas as pd

from bs4 import BeautifulSoup



import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Dresses = pd.read_csv('/kaggle/input/Dresses.csv').drop('Unnamed: 0' , axis=1)
Dresses.head(1)
#Investigate missing values

Dresses.info()
Dresses.isna().sum()
Dresses.describe()
Dresses.hist(figsize = (15,10) , color='thistle');
brand_price = Dresses[['brand' , 'price' , 'reduction_price' , 'reduction_percentage']].groupby(['brand']).mean().reset_index()

brand_price.sort_values(by=['price'], ascending=False)
brand_price.sort_values(by=['reduction_price'], ascending=False)
brand_price.sort_values(by=['reduction_percentage'], ascending=False)
fig , ax = plt.subplots( figsize=(15,5))

plt.xticks(rotation='vertical')

ax = sns.countplot(Dresses.brand)
fig , ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation='vertical')

ax = plt.bar(brand_price.brand , brand_price.price , color='thistle')
fig , ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation='vertical')

ax = plt.bar(brand_price.brand , brand_price.reduction_price , color='thistle')
fig , ax = plt.subplots(figsize=(15,5))

plt.xticks(rotation='vertical')

ax = plt.bar(brand_price.brand , brand_price.reduction_percentage , color='thistle')
sns.countplot(Dresses.dress_length)
fig , ax = plt.subplots( figsize=(10,4))

plt.xticks(rotation='vertical')

ax = sns.countplot(Dresses.sleeve_type)
comment_words = ' '

stopwords = set(STOPWORDS) 

  

# iterate through the csv file 

for val in Dresses.brand: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 