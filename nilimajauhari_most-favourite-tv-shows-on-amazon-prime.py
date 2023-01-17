# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the data set

tv_shows = pd.read_csv("/kaggle/input/amazon-prime-tv-shows/Prime TV Shows Data set.csv",encoding="iso-8859-1")
# Let us look at the shape of the data

tv_shows.shape
# Visualizing the first few rows of the data set

tv_shows.head(3)
# Let us take a look at the data types of each variable

tv_shows.dtypes
# Looking at the age of viewers

import seaborn as sns

sns.set(style="darkgrid")

ax = sns.countplot(x = "Age of viewers", data = tv_shows)
# Let us take a look at the languages in which TV shows are being offered

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y = "Language", data = tv_shows)
# Let us take a look at the genres of the TV shows

f, ax = plt.subplots(figsize=(7, 10))

sns.countplot(y = "Genre", data = tv_shows)
# Let us take a look at how many seasons of the show are available on Prime

f, ax = plt.subplots(figsize=(7, 10))

sns.countplot(y = "No of seasons available", data = tv_shows)
# Let us now check for missing values

tv_shows.isnull().sum()
# Checking for outliers

sns.boxplot(x ='Age of viewers',y = 'IMDb rating',data = tv_shows,palette ='rainbow')
# Replace using median 

median = tv_shows['IMDb rating'].median()

tv_shows['IMDb rating'].fillna(median, inplace=True)
tv_shows.isnull().sum()
# Let us take a look at the top 20 high rated shows on Amazon Prime

tv_shows.sort_values(by = "IMDb rating", ascending = False).head(20)
# Let us now take a look at 20 worst rated shows

tv_shows.sort_values(by = "IMDb rating", ascending = True).head(20)
top_english = tv_shows[tv_shows['Language'] == 'English'].sort_values(by = 'IMDb rating',ascending = False)

#Top 10 TV shows in english

top_english.head(10)
# Top 10 TV shows in Hindi

top_hindi = tv_shows[tv_shows['Language'] == 'Hindi'].sort_values(by = 'IMDb rating',ascending = False)

#Top 10 TV shows in hindi

top_hindi.head(10)
# Top 10 TV shows in the genre: 'Drama'

top_drama = tv_shows[tv_shows['Genre'] == 'Drama'].sort_values(by = 'IMDb rating',ascending = False)

#Top 10 TV shows in drama

top_drama.head(10)
# Top 10 TV shows in the genre: 'Comedy'

top_comedy = tv_shows[tv_shows['Genre'] == 'Comedy'].sort_values(by = 'IMDb rating',ascending = False)

#Top 10 TV shows in comedy

top_comedy.head(10)
# Top 10 TV shows in the genre: 'Kids'

top_kids = tv_shows[tv_shows['Genre'] == 'Kids'].sort_values(by = 'IMDb rating',ascending = False)

#Top 10 TV shows for kids

top_kids.head(10)
# Let us now look at the top 10 best shows released this year which are a must watch

top10 = tv_shows[tv_shows['Year of release'] == 2020].sort_values(by ='IMDb rating',ascending = False)

#Top 10 TV shows of 2020

top10.head(10)
# Visualizing the most used words in the names of the TV shows

common_words = tv_shows['Name of the show'] 

word_cloud = WordCloud(width = 1000,

                       height = 800,

                       colormap = 'GnBu', 

                       margin = 0,

                       max_words = 200,  

                       min_word_length = 4,

                       max_font_size = 120, min_font_size = 15,  

                       background_color = "white").generate(" ".join(common_words))



plt.figure(figsize = (10, 15))

plt.imshow(word_cloud, interpolation = "gaussian")

plt.axis("off")

plt.show()