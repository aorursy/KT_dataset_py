# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/tmdb_5000_movies.csv")
data.info()
data.corr()
f,ax = plt.subplots(figsize=(16, 16))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
data.head()
data.columns
data.describe()
dataFrame=data[['popularity']]
print(type(dataFrame))
print('')
series=data['popularity']
print(type(series))
data.plot(kind='scatter', x='vote_average', y='popularity',alpha = 0.5,color = 'red')
plt.xlabel('Vote Average')              # label = name of label
plt.ylabel('Popularity')
plt.title('Vote Average and Popularity Scatter Plot')  
data.popularity.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data[(data['popularity']>150.0) & (data['vote_average']>7.5)]