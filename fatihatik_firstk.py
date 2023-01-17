# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
datac=pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")

datam=pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
datac.info()

datam.info()
display(datam.corr())
f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(datam.corr(), annot=True, linewidth=1, fmt='.1f', ax=ax)

plt.title("Movie Datas")

plt.show()
datam.head(10)
datam.columns

datam['budget'].plot(kind='line', color='red', label='Bütçe',linewidth=0.5,alpha=0.5,grid=True, figsize=(10,10),linestyle=':')

datam['revenue'].plot(kind='line', color='green', label='Popülerlik', linewidth=0.5,alpha=0.5,grid=True,figsize=(10,10), linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('Bütçe')

plt.ylabel('Gelir')

plt.title('Bütçe-Gelir')

plt.show()
datam.columns
datam.plot(kind='scatter', x='budget' , y='revenue' ,alpha=0.7, color='blue' , figsize=(10,10))

plt.xlabel('Bütçe')

plt.ylabel('Gelir')

plt.title('Bütçe-Gelir')

plt.show()
datam['runtime'].plot(kind='hist', bins=50, figsize=(10,10))

plt.show()
datam[datam['runtime']<50]