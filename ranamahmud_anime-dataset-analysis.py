# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
anime = pd.read_csv("../input/anime-dataset/anime_data.csv")

user = pd.read_csv("../input/anime-dataset/user_data.csv")

# shape of the data

anime.shape
# first few rows

anime.head()
# descriptive statisitcs of numeric column

anime.describe()
anime['type'].value_counts()
# anime type plot

anime['type'].value_counts().plot(kind='bar')

plt.title("Type of Anime")

plt.xlabel("Type")

plt.ylabel("Frequency")
anime['status'].value_counts()
# anime status

anime['status'].value_counts().plot(kind='bar')

plt.title("Status of Anime")

plt.xlabel("Type")

plt.ylabel("Frequency")
# possibility of droping status
# anime source

anime['source'].value_counts().plot(kind='bar')

plt.title("Source of Anime")

plt.xlabel("Type")

plt.ylabel("Frequency")
# distribution of Anime Eposodes

data = anime['episodes']

binwidth = 10

anime['episodes'].hist(bins=range(min(data), max(data) + binwidth, binwidth))

plt.xlim(0,150)

plt.title("Distribution of Anime Episode Number")

plt.ylabel("Frequency")

plt.xlabel("Number of Episode")
plt.boxplot(anime['episodes'])

plt.title("Boxplot of Anime Episode Number")

plt.ylabel("Number of Episode")
# popularity of anime

data = anime['popularity']

binwidth = 200

anime['popularity'].hist(bins=range(min(data), max(data) + binwidth, binwidth))

# plt.xlim(0,150)

plt.title("Distribution of Anime Popularity")

plt.ylabel("Frequency")

plt.xlabel("Popularity")
anime['rank'].nunique()
# rank of anime

data = anime['rank']

# binwidth = 200

data.hist()

# plt.xlim(0,150)

plt.title("Distribution of Anime Rank")

plt.ylabel("Frequency")

plt.xlabel("Popularity")
# distribution of anime score

data = anime['score']

# binwidth = 200

data.hist()

# plt.xlim(0,150)

plt.title("Distribution of Anime Score")

plt.ylabel("Frequency")

plt.xlabel("Score")
plt.boxplot(anime['score'])

plt.title("Boxplot of Anime Score")

plt.ylabel("Number of Episode")
# distribution of anime score

data = anime['scored_by']

binwidth = 200

data.hist(bins=range(int(min(data)), int(max(data)) + binwidth, binwidth))

plt.xlim(0,20000)

plt.title("Distribution of Anime by Number of Score")

plt.ylabel("Frequency")

plt.xlabel("Number of Score")
plt.boxplot(anime['scored_by'])

plt.title("Boxplot of Anime  by Number of Score")

plt.ylabel("Number of Scorer")
prem_df = anime['premiered'].value_counts()

prem_df.head()
# premiered

prem_df[0:5].plot(kind='bar')

plt.title("Top 5 Time by Number of Anime Premiered")
# data engineering 

year_season = anime['premiered'].str.split(expand=True)

year_season.head()

year_season.columns = ['season','year']

year_season.head()
anime.drop('premiered',axis = 1, inplace = True)
anime = anime.join(year_season)
# Anime by Season Premiered

anime['season'].value_counts().plot(kind='bar')

plt.title("Anime Premiered by Season")

plt.xlabel("Season")

plt.ylabel("Number of Anime")