# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)

from wordcloud import WordCloud

from geopy.geocoders import Nominatim

from folium.plugins import HeatMap

import folium

from tqdm import tqdm

import re

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim

from collections import Counter

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import matplotlib.colors as mcolors

# Any results you write to the current directory are saved as output.
resData = pd.read_csv('../input/zomato.csv')

resData.head()
print('Data types')

print(resData.dtypes)

print('Info')

print(resData.info())
plt.figure(figsize=(10,7))

chains=resData['name'].value_counts()[:10]

sns.barplot(x=chains,y=chains.index,palette='deep')

plt.title("Most popular restaurants chains in Bangaluru")

plt.xlabel("Number of outlets")
x=resData['online_order'].value_counts()

print(x)

labels=['Yes','No']

colors = ['#FEBFB3', '#E1396C']

#trace=plt.pie(x,labels=labels,autopct='%1.1f%%',shadow=True,textprops="values")

#to make values and percent both appear in the pie chart

#try not to use this function as it involves lot of calculation using matplotlib. just use sseaborn to make life easier

def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct



plt.pie(x, labels=labels, autopct=make_autopct(x),colors=colors,shadow=True)



print(type(x))

#trace=go.Pie(labels=x.index,values=x)

#layout=go.Layout(title="Accepting vs not accepting online orders",width=500,height=500)

#fig=go.Figure(data=[trace],layout=layout)

#py.iplot(fig, filename='pie_chart_subplots')
x=resData['book_table'].value_counts()

#textinfo="value" is used in go.Pie to write values. if we remove textinfo it changes to percent

#in text values add value in the array to display both values and percent in the graph

trace=go.Pie(labels=x.index,values=x,text=x,

            marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

layout=go.Layout(title="Table booking",width=500,height=500)

fig=go.Figure(data=[trace],layout=layout)

py.iplot(fig, filename='pie_chart_subplots')
plt.figure(figsize=(6,5))

restrntWithRating=resData['rate'].dropna()



rating=resData['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()

sns.distplot(rating,bins=20)
costDist= resData[['rate','approx_cost(for two people)','online_order']].dropna()

costDist['rate']= costDist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)

#use pandas.to_numeric to convert simple series data into integer and use .replace(',','') to convert in case of single string





costDist['approx_cost(for two people)']= costDist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))

costDist['approx_cost(for two people)'].head()
plt.figure(figsize=(15,7))

sns.stripplot(x="rate",y='approx_cost(for two people)',data=costDist)

plt.show()
plt.figure(figsize=(6,6))



sns.distplot(costDist['approx_cost(for two people)'])

plt.show()