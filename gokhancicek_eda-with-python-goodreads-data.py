# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False)

df.head()
df.tail()
df.info()
df.columns
df.rename(columns={'  num_pages':'num_pages'}, inplace=True)
df.shape
df.nunique()
df.duplicated().sum()
df.describe()
df.corr()
f, ax = plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(), annot=True, linewidths=0.5, fmt='.2f', ax=ax)
df.plot(x='publication_date',y='average_rating', color='r', label='Ratings', linewidth=1, alpha=0.7, grid=True, linestyle=':')

#df.plot(kind='line', color='g', label='Reviews', linewidth=1, alpha=0.7, grid=True, linestyle=':')

plt.xlabel('xaxsis')

plt.ylabel('yaxsis')

plt.title('Ratings vs. Reviews')

plt.show()



df.plot(kind='scatter', x='ratings_count',y='text_reviews_count', alpha=.7, color='b')

plt.xlabel('Ratings')

plt.ylabel('Reviews')

plt.title('Scatter')

plt.show()
df.average_rating.plot(kind='hist', bins=100, figsize=(8,8))

plt.xlabel('Average Rating')

plt.title('Average Rating Frequency');
x = df['num_pages']>2000

df[x]
df[np.logical_and(df['average_rating']>4.5, df['ratings_count']>5000)]
for index,value in df[['title']][500:505].iterrows():

    print(index," : ",value)