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
data =pd.read_csv('../input/tmdb_5000_movies.csv') #import data
data.info() 
data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
data.plot(kind='scatter',x='revenue',y='vote_count',alpha=0.5,color='red')
plt.xlabel=('Revenue')
plt.ylabel=('Vote Count')
plt.title('Revenue Vote Count Scatter Plot')
plt.show()
data.plot(kind='scatter',x='runtime',y='vote_count',alpha=0.5,color='red')
plt.xlabel=('Runtime')
plt.ylabel=('Vote Count')
plt.title('Runtime Vote Count Scatter Plot')
plt.show()
data.vote_average.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data_frame =data[['genres']]
print (data_frame)
x = data['popularity']>100    
data[x]
data[(data['budget']>200000000)&(data['popularity']>100)&(data['revenue']>200000000)]