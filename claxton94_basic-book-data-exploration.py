# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bkdf = pd.read_csv('/kaggle/input/books.csv',error_bad_lines = False)
bkdf.info()


x = len(list(set(bkdf['average_rating'])))

print(x)
bkdf.reindex(bkdf.iloc[:,0])

bkdf.drop('bookID',axis=1,inplace=True)

(bkdf.groupby('average_rating').count()).loc[:,'ratings_count']
bkdf.plot(y = 'average_rating', kind = 'hist',bins = 50)

plt.xlabel('Average Rating')

plt.ylabel('Count')
bkdf.plot(x = 'average_rating',y='ratings_count', kind = 'scatter')

bkdf.plot(y = 'average_rating', kind = 'box')
bkdf2 = bkdf[bkdf['average_rating'] == 5]

bkdf2.head()