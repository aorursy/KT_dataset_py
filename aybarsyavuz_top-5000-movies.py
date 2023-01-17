# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.info()
data.corr()
data.head(10)
data.columns
data.tail()
data.budget.plot(kind = 'line', color = 'red',label = 'budget', linewidth=1, alpha=1, grid=True, figsize = (30,30))
data.revenue.plot(kind = 'line', color = 'green', label = 'revenue', linewidth=1, alpha=0.5, grid=True, figsize = (30,30))
plt.legend(loc = 'upper left')
plt.xlabel('budget')
plt.ylabel('revenue')
plt.title('Budget vs Revenue')
plt.show()
data.plot(kind = 'scatter', x = 'budget', y = 'vote_average', alpha = 1, color = 'green', figsize = (30,30))
plt.xlabel('budget')
plt.ylabel('vote_average')
plt.title('Comparison')
plt.show()
data.vote_average.plot(kind = 'hist', bins= 100, figsize = (30,30))
plt.xlabel('vote rating')
plt.show()
va = data['vote_average'] > 8.0
data[va]
type(data)
data1 = data.drop(['genres','homepage','id','keywords','runtime','status','tagline','vote_count','overview','production_companies','production_countries','spoken_languages'],axis=1)
data2 = data1[va]
data2
data2.sort_values(by=['budget'])
data2.sort_values(by = 'revenue')
data2.sort_values(by='vote_average')
