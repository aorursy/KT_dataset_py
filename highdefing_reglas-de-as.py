# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for visualizations

import matplotlib.pyplot as plt

import squarify

import seaborn as sns

plt.style.use('fivethirtyeight')



# for analysis

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading the dataset



data_anime = pd.read_csv('../input/anime-recommendations-database/anime.csv', header = None)

data_rating = pd.read_csv('../input/anime-recommendations-database/rating.csv', header = None)

# let's check the shape of the dataset

print("data_anime:",data_anime.shape)

print("data_rating:",data_rating.shape)
data_anime.head()


data_anime_1 = data_anime



#put column names

data_anime_1.columns = data_anime_1.iloc[0] #set first row as column names

data_anime_1 = data_anime_1.drop(data_anime_1.index[0]) #delete first row



data_anime_1.head()
data_rating.head()


data_rating_1 = data_rating



#put column names

data_rating_1.columns = data_rating_1.iloc[0] #set first row as column names

data_rating_1 = data_rating_1.drop(data_rating_1.index[0]) #delete first row



data_rating_1.head()
data_anime_1.tail()
data_rating_1.tail()
# checking the random entries in the data_anime



data_anime_1.sample(10)
# checking the random entries in the data_rating



data_rating_1.sample(10)
#describe data_anime

data_anime_1.describe()
# describe data_rating

data_rating_1.describe()
data_anime_1_members = data_anime_1.sort_values(by=['members'],ascending=False)

data_anime_1_members
# name of animes with most members

import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud



plt.rcParams['figure.figsize'] = (15, 15)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(data_anime_1['genre']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Items',fontsize = 20)

plt.show()
data_anime_1_value_counts = data_anime_1.name.value_counts()



data_anime_1_value_counts.sort_values(ascending=False)
data_anime_1.loc[data_anime_1.genre == 'object']
data_anime_1[data_anime_1['genre'].str.contains("object")]