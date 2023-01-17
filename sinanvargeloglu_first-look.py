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
data1 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

data2 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
data1.info()
data2.info()
data1.columns
data1.head(5)
data2.columns
data2.head(5)
data2.corr()
data2.describe()
plt.subplots(figsize=(15,12))

plt.scatter(data2.revenue,data2.vote_count,color='red')

plt.ylabel('Vote Count')             

plt.xlabel('Revenue')

plt.title('Revenue - Vote Count Plot')
plt.subplots(figsize=(15,12))

plt.scatter(data2.revenue,data2.popularity,color='green')

plt.xlabel('Revenue')             

plt.ylabel('Popularity')

plt.title('Revenue - Popularity Plot')
plt.subplots(figsize=(15,12))

plt.scatter(data2.vote_count,data2.popularity,color='blue')

plt.xlabel('Vote Count')             

plt.ylabel('Popularity')

plt.title('Vote Count - Popularity Plot')
plt.subplots(figsize=(15,12))

plt.scatter(data2.revenue,data2.budget,color='purple')

plt.xlabel('Revenue')             

plt.ylabel('Budget')

plt.title('Revenue - Budget Plot')
plt.subplots(figsize=(15,8))

plt.hist(data2.vote_average,bins = 100)

plt.ylabel('Frequency')

plt.xlabel('Vote Average')

plt.title('Vote Average Graphic Distribution')

plt.show()
plt.subplots(figsize=(10,8))

plt.scatter(data2.vote_average,data2.budget,color='red')

plt.xlabel('Vote Average')             

plt.ylabel('Budget')

plt.title('Vote Average - Budget Plot')

plt.show()
x = data2['budget']>3000000

data2[x]

#plt.scatter(data2.vote_average,x,color='blue')

plt.subplots(figsize=(20,16))

plt.scatter(data2.vote_average,data2.revenue,color='green')

plt.xlabel('Vote Average')             

plt.ylabel('Revenue')

plt.title('Vote Average - Revenue')

plt.show()