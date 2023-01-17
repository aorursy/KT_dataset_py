%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import sqlite3

# Connect database
db = sqlite3.connect('../input/database.sqlite')
# Query brands from table and remove empty brands. Stores as Pandas DataFrame
query = pd.read_sql_query('SELECT brands FROM FoodFacts WHERE brands != ""', db)
print(query.head())
print(query.tail())
# WordCloud must contain input in one long string
brand_string = ''

# Extract DataFrame into string (adding spaces between each instance)
for brand in range(len(query.brands)):
    brand_string += " " + str(query.brands[brand])

#print(brand_string)
# Create wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

wordcloud = WordCloud(stopwords=STOPWORDS, 
                      background_color='white')
wordcloud.generate(brand_string)

plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
# Query additives from table and remove empty additives. Stores as Pandas DataFrame
query = pd.read_sql_query('SELECT additives FROM FoodFacts WHERE additives != ""', db)
print(query.head())
print(query.tail())
# To check if words are in the english dictionary. Credit: http://stackoverflow.com/questions/29099621/how-to-find-out-wether-a-word-exists-in-english-using-nltk
import nltk
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

# WordCloud must contain input in one long string
additive_string = ''

# Extract DataFrame into string (adding spaces between each instance)
for additive in range(len(query.additives)):
    additive_string += " " + str(query.additives[additive])

# Clean string to leave only letters
additive_string = ''.join([char for char in additive_string if (char.isalpha() or char == ' ')])

# Leave only words in english language (eliminates need for further cleaning)
temp_list = additive_string.split(' ')
additive_string = ' '.join([word for word in temp_list if word in english_vocab])

# print(additive_string)
# Create wordcloud
wordcloud = WordCloud(stopwords=STOPWORDS, 
                      background_color='white')
wordcloud.generate(additive_string)

plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
# Query origins from table and remove empty origins. Stores as Pandas DataFrame
query = pd.read_sql_query('SELECT origins FROM FoodFacts WHERE origins !="" ', db)
print(query.head())
print(query.tail())
# WordCloud must contain input in one long string
origin_string = ''

# Extract DataFrame into string (adding spaces between each instance)
for i in range(len(query.origins)):
    origin_string += " " + str(query.origins[i])

# print(origin_string)
# Create wordcloud
wordcloud = WordCloud(stopwords=STOPWORDS, 
                      background_color='white')
wordcloud.generate(origin_string)

plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
