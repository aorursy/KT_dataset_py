# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
reviews = pd.read_csv('../input/winemag-data_first150k.csv',index_col = 0)
reviews.head()
reviews.province.value_counts().head(10).plot.bar()
(reviews.province.value_counts().sort_values(ascending=False).head(10)/len(reviews)).plot.bar()
reviews['points'].value_counts().sort_index().plot.bar()
reviews['points'].value_counts().sort_index().plot.line()
reviews['points'].value_counts().sort_index().plot.area()
reviews[reviews['price']<200]['price'].plot.hist()
reviews['price'].plot.hist()
reviews[reviews['price']>1500]
reviews['points'].plot.hist()
reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price',y='points')
reviews[reviews['price']<100].plot.scatter(x='price',y='points')
reviews[reviews['price']<100].plot.hexbin(x='price',y='points',gridsize=15)