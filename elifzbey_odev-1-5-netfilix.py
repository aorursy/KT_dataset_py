# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
data.head()
data.info()
data.columns
data.corr()
data.plot(kind='bar',x='release_year',y='show_id')
data.plot(kind='scatter', x='release_year', y='show_id',alpha=0.5,color= 'red')

plt.xlabel('release_year')

plt.ylabel('show_id')

plt.title('ScatterPlot')
data1=data['release_year']>2017

data[data1]

data.release_year.plot(kind='hist',bins=100,figsize=(10,10))

plt.show()
data.assign(release_year = 1).groupby(

  ['release_year','country']).size().to_frame().unstack().plot(kind='bar',stacked=True,legend=False)