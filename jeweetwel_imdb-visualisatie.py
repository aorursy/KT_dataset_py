# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data uploaden

imdb_data_filepath = "../input/imdb-data/data.tsv"

imdb_data = pd.read_csv(imdb_data_filepath, index_col = 'tconst', sep = '\t', low_memory=False)



imdb_ratings_filepath = "../input/imdb-data/data.ratings.tsv"

imdb_ratings = pd.read_csv(imdb_ratings_filepath, index_col = 'tconst', sep = '\t', low_memory=False)
#Tabellen samenvoegen tot een tabel

imdb = imdb_data.join(imdb_ratings)

imdb_selectie = imdb[(imdb.startYear > 1895) & (imdb.startYear < 2020)]
#Voor de lol uitgeprobeerd hoe het eruit zag

#plt.figure(figsize=(20,23))

#sns.scatterplot(x=imdb['startYear'], y=imdb['averageRating'])
#Tellen hoeveel films per jaar en het berekenen van de gemiddelde beoordeling per jaar

yearCount = imdb.groupby('startYear').startYear.count()

yearRating = imdb.groupby('startYear').averageRating.mean()
#De data croppen (eerst teste ik hier om te kijken wat er weg moest)

yearCount.iloc[14:138]

yearCount2 = yearCount[14:138]

yearRating2 = yearRating[60:138]
#Code om de grafieken te maken

plt.figure(figsize=(25,6))

plt.title("Average rating over the years")

sns.lineplot(data=yearRating2)



plt.figure(figsize=(25,6))

plt.title("Number of films/series over the years")

sns.lineplot(data=yearCount2)