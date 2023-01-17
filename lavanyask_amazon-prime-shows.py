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

import missingno as msno

import seaborn as sns
prime_df = pd.read_csv('../input/amazon-prime-tv-shows/Prime TV Shows Data set.csv', encoding = 'iso-8859-1')
prime_df
prime_df.info()
# looking for null values

prime_df.isnull().sum()
msno.matrix(prime_df);
# percentage of missing values in IMDb rating

(prime_df['IMDb rating'].isnull().sum()/prime_df.shape[0])*100
prime_df['Language'].value_counts()
nadf = prime_df[prime_df['IMDb rating'].isnull()]
# the distribution of na's per language

nadf['Language'].value_counts()
# number of na's per language / number of values per language

nadf['Language'].value_counts() / prime_df['Language'].value_counts()
# looking at the data types

prime_df.dtypes
# looking at the first 10 entries of the dataset

prime_df.head(10)
# drop S.no. column

prime_df.drop(['S.no.'],axis = 1, inplace = True)

prime_df
sns.set_style('whitegrid')
#plotting the ratings by year of release

plt.figure(figsize = (16,7))

sns.barplot(x = 'Year of release', y = 'IMDb rating', data = prime_df.dropna(axis = 0, subset = ['IMDb rating']));
# plotting number of shows per year

plt.figure(figsize = (16,7))

sns.countplot(y = 'Year of release', data = prime_df);
# plotting the shows by number of seasons

plt.figure(figsize = (12,6))

sns.countplot(y = 'No of seasons available', data = prime_df, palette="Blues_d");
# plotting the shows by language

plt.figure(figsize = (12,6))

sns.countplot(y = 'Language', data = prime_df);
# plotting the shows by genres

plt.figure(figsize = (18,9))

sns.countplot(y = 'Genre', data = prime_df);
# plotting age of viewers

sns.countplot(x = 'Age of viewers', data = prime_df);
prime_df.sort_values(by = 'IMDb rating', ascending = False).head(10)
(prime_df.sort_values(by = 'IMDb rating', ascending = False).head(30)).groupby(['Genre']).count()
prime_df.sort_values(by = 'IMDb rating').head(10)
(prime_df.sort_values(by = 'IMDb rating').head(30)).groupby(['Genre']).count()
prime_df[prime_df['Year of release'] == 2020].sort_values(by = 'IMDb rating', ascending = False).head(10)
prime_df[prime_df['Language'] == 'English'].sort_values(by = 'IMDb rating', ascending = False).head(5)
prime_df[prime_df['Language'] == 'Hindi'].sort_values(by = 'IMDb rating', ascending = False).head(5)
prime_df[prime_df['No of seasons available'] == prime_df['No of seasons available'].max()]
prime_df[prime_df['No of seasons available'] >= 5].sort_values(by = 'IMDb rating', ascending = False).head(10)
prime_df[prime_df['Year of release'] == prime_df['Year of release'].min()]