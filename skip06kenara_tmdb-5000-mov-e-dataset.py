# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/tmdb_5000_movies.csv")

data.corr()
data.info()

data.corr()

f,ax = plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)

plt.show()
data.head(10)

data.tail()
data.columns
data.budget.plot(kind = 'line' , color = 'g', label = 'budget', linewidth=1,alpha = 0.5, grid = True,linestyle = ':')

data.revenue.plot(color = 'r', label = 'revenue', linewidth=1, alpha = 0.5, grid = True, linestyle= ':')

plt.legend(loc='upper right')

plt.xlabel('revenue')

plt.ylabel('budget')

plt.title('Line Movie Plot')

data.plot(kind = 'scatter', x = 'revenue', y = 'popularity',alpha = 0.5, color = 'red')

plt.xlabel('revnue')

plt.ylabel('popularity')

plt.title('scatter plot movie')
data.popularity.plot(kind = 'hist' , bins = 50, figsize= (15,15))

plt.show()



data.revenue.plot(kind = 'hist', bins= 40, figsize = (12,12))

plt.show()
plt.clf()
dictionary = {'spain' : 'madrid', 'usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())

dictionary['spain'] = "barcelona"

print(dictionary)

dictionary['france'] = "paris"

print(dictionary)

del dictionary['spain']

print(dictionary)

print('france' in dictionary)

dictionary.clear()

print(dictionary)
series = data['popularity']

print(type(series))

data_frame = data[['revenue']]

print(type(data_frame))

data[np.logical_and(data['popularity']>2000, data['revenue']>2500)]
