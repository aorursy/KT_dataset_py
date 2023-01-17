# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# read data 
data = pd.read_csv('../input/tmdb_5000_movies.csv')
#look up for data
data.info()
# columns in data
data.columns
# look up correlation for data 
data.corr()
# visual correlation
sns.heatmap(data.corr(),annot=True, linewidths=1)
plt.show()
print(data.head(10))

data.vote_count.plot(kind = 'line',color='blue',label='Vote Count', linewidth=1,alpha=0.5, grid=True)
plt.legend(loc='upper right')
plt.show()
data.plot(kind='scatter', x='vote_count', y='vote_average',alpha = 0.5,color = 'red')
plt.xlabel('Count Vote')
plt.ylabel('Average Vote')
plt.show()
#Histogram
data.vote_average.plot(kind = 'hist',bins = 50)
plt.show()
print(data.columns)


#Filtering
filter_average_vote_greater_than_7 = data.vote_average > 7
filter_vote_count_greater_than_5000 = data.vote_count > 5000

data1 = data[filter_average_vote_greater_than_7]
sns.heatmap(data1.corr(), annot=True)
plt.show()


data2 = data[filter_vote_count_greater_than_5000]
plt.clf()
sns.heatmap(data2.corr(), annot=True)
plt.show()


data3 = data[filter_average_vote_greater_than_7 & filter_vote_count_greater_than_5000]
plt.clf()
sns.heatmap(data3.corr(), annot=True)
plt.show()
# sort by avarage vote
sorted = data3.sort_values('vote_average', ascending=False)

# lets print these movies names and vote average that filtered and sorted by average vote
for index,movie in sorted.iterrows():
    print("Movie name : ", movie.title,", average vote : ", movie.vote_average)
    #print(movie)
