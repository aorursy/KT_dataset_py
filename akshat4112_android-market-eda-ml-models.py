import numpy as np 

import pandas as pd 

import os

import seaborn as sns

import nltk

import matplotlib.pyplot as plt 

import PIL



# NLTK

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from nltk.classify import SklearnClassifier

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

# nltk.download('wordnet')

# nltk.download('stopwords')



# Other

import re

import string

from wordcloud import WordCloud

from IPython.display import display

import base64

import string

import re

from textblob import TextBlob

from collections import Counter

from time import time

from subprocess import check_output



# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords

from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split # function for splitting data to train and test sets

from sklearn.manifold import TSNE



#import gensim

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS



plt.style.use('ggplot') 
data = pd.read_csv("../input/googleplaystore.csv")
data.head()
data.drop_duplicates(subset='App', inplace=True)

data = data[data['Android Ver'] != np.nan]

data = data[data['Android Ver'] != 'NaN']

data = data[data['Installs'] != 'Free']

data = data[data['Installs'] != 'Paid']
def type_cat(types):

    if types == 'Free':

        return 0

    else:

        return 1
data['Type'] = data['Type'].map(type_cat)
data.sample(7)
len(data)
rev_data = pd.read_csv("../input/googleplaystore_user_reviews.csv")
rev_data.head()
data.columns
data.dtypes
genresString = data["Genres"]

genresVal = data["Genres"].unique()

genresValCount = len(genresVal)

genres_dict = {}

for i in range(0,genresValCount):

    genres_dict[genresVal[i]] = i

data["Genres"] = data["Genres"].map(genres_dict).astype(int)
CategoryString = data["Category"]

categoryVal = data["Category"].unique()

categoryValCount = len(categoryVal)

category_dict = {}

for i in range(0,categoryValCount):

    category_dict[categoryVal[i]] = i

data["Category"] = data["Category"].map(category_dict).astype(int)
data['Content Rating'] = data['Content Rating'].map({'Everyone':0,'Teen':1,'Everyone 10+':2,'Mature 17+':3,

                                                     'Adults only 18+':4}).astype(float)
data['Reviews'] = [ float(i.split('M')[0]) if 'M'in i  else float(i) for i in data['Reviews']]
data["Size"] = [ float(i.split('M')[0]) if 'M' in i else float(0) for i in data["Size"]  ]
data['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in data['Price'] ] 
data["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in data["Installs"] ]
data["Rating"] = data.groupby("Category")["Rating"].transform(lambda x: x.fillna(x.mean()))

data["Content Rating"] = data[["Content Rating"]].fillna(method="ffill")
data["Type"] = data[["Type"]].fillna(method="ffill")
rev_data.columns
data.shape
rev_data.shape
data.Rating.describe()
data.Installs.describe()
data.Reviews.describe()
rev_data.Sentiment_Polarity.describe()
rev_data.Sentiment_Subjectivity.describe()
data.Category.value_counts().head()
data.Rating.value_counts().head()
data.Size.value_counts().head()
data.Installs.value_counts().head()
data.Price.value_counts().head()
data.Genres.value_counts().head()
rev_data.Sentiment.value_counts()
rev_data.Sentiment_Polarity.value_counts().head()
rev_data.Sentiment_Subjectivity.value_counts().head()
data.Category.value_counts().plot(kind='barh',figsize=(20,12))
data['Content Rating'].value_counts().plot(kind='barh')
data['Genres'].value_counts().plot(kind='barh',figsize=(25,30))
data['Rating'].value_counts().plot(kind = 'barh', figsize=(10,10))
data['Installs'].value_counts().plot(kind = 'barh', figsize = (10,10))
data['Type'].value_counts().plot(kind = 'bar', figsize = (5,5))
data['Android Ver'].value_counts().plot(kind = 'barh', figsize = (10, 10))
rev_data['Sentiment'].value_counts().plot(kind = 'bar', figsize = (10,10))
# rev_data['Sentiment_Subjectivity'].value_counts().plot(kind = 'bar', figsize = (10,10))
# rev_data['Sentiment_Polarity'].value_counts().plot(kind = 'bar', figsize = (10,10))
plt.scatter(data['Installs'], data['Rating'], color='b')

plt.xlabel('Installs')

plt.ylabel('Ratings')

plt.show()
plt.scatter(data['Installs'], data['Category'], color='b')

plt.xlabel('Installs')

plt.ylabel('Category')

plt.show()
plt.scatter(data['Installs'], data['Size'], color='b')

plt.xlabel('Installs')

plt.ylabel('Size')

plt.show()
plt.scatter(data['Installs'], data['Genres'], color='b')

plt.xlabel('Installs')

plt.ylabel('Genres')

plt.show()
plt.scatter(data['Installs'], data['Reviews'], color='b')

plt.xlabel('Installs')

plt.ylabel('Reviews')

plt.show()
plt.scatter(data['Installs'], data['Type'], color='b')

plt.xlabel('Installs')

plt.ylabel('Type')

plt.show()
plt.scatter(data['Installs'], data['Price'], color='b')

plt.xlabel('Installs')

plt.ylabel('Price')

plt.show()
plt.scatter(data['Installs'], data['Content Rating'], color='b')

plt.xlabel('Installs')

plt.ylabel('Content Rating')

plt.show()
plt.scatter(data['Category'], data['Rating'], color='b')

plt.xlabel('Category')

plt.ylabel('Rating')

plt.show()
plt.scatter(data['Category'], data['Reviews'], color='b')

plt.xlabel('Category')

plt.ylabel('Reviews')

plt.show()
plt.scatter(data['Category'], data['Size'], color='b')

plt.xlabel('Category')

plt.ylabel('Size')

plt.show()
plt.scatter(data['Category'], data['Type'], color='b')

plt.xlabel('Category')

plt.ylabel('Type')

plt.show()
plt.scatter(data['Category'], data['Price'], color='b')

plt.xlabel('Category')

plt.ylabel('Price')

plt.show()
plt.scatter(data['Category'], data['Content Rating'], color='b')

plt.xlabel('Category')

plt.ylabel('Content Rating')

plt.show()
plt.scatter(data['Rating'], data['Reviews'], color='b')

plt.xlabel('Rating')

plt.ylabel('Reviews')

plt.show()
plt.scatter(data['Rating'], data['Size'], color='b')

plt.xlabel('Rating')

plt.ylabel('Size')

plt.show()
plt.scatter(data['Type'], data['Rating'], color='b')

plt.xlabel('Type')

plt.ylabel('Rating')

plt.show()
plt.scatter(data['Price'], data['Rating'], color='b')

plt.xlabel('Price')

plt.ylabel('Rating')

plt.show()
plt.scatter(data['Reviews'], data['Size'], color='b')

plt.xlabel('Reviews')

plt.ylabel('Size')

plt.show()
plt.scatter(data['Reviews'], data['Type'], color='b')

plt.xlabel('Reviews')

plt.ylabel('Type')

plt.show()
plt.scatter(data['Reviews'], data['Price'], color='b')

plt.xlabel('Reviews')

plt.ylabel('Price')

plt.show()
plt.scatter(data['Reviews'], data['Content Rating'], color='b')

plt.xlabel('Reviews')

plt.ylabel('Content Rating')

plt.show()
plt.scatter(data['Reviews'], data['Genres'], color='b')

plt.xlabel('Reviews')

plt.ylabel('Genres')

plt.show()
plt.scatter(data['Size'], data['Genres'], color='b')

plt.xlabel('Size')

plt.ylabel('Genres')

plt.show()
plt.scatter(data['Size'], data['Type'], color='b')

plt.xlabel('Size')

plt.ylabel('Type')

plt.show()
plt.scatter(data['Genres'], data['Type'], color='b')

plt.xlabel('Genres')

plt.ylabel('Type')

plt.show()
plt.scatter(data['Price'], data['Genres'], color='b')

plt.xlabel('Price')

plt.ylabel('Genres')

plt.show()