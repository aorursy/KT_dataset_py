# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/amazon-top-50-bestselling-books-2009-2019/bestsellers with categories.csv', engine='python')

df.head()
df.info()
df['Genre'].value_counts()
df['Genre'].value_counts(normalize=True)
# Top rated books (Revieved by at least 3000 people)

data_order = df.groupby(['Name', 'Author', 'Genre'], as_index=False)[['User Rating', 'Reviews']].mean()

data_order = data_order[data_order['Reviews']>3000]

data_order = data_order.sort_values('User Rating', ascending=False).head(20)

data_order
# Weighted rating

m = min(df['Reviews'])

C = df['User Rating'].mean()

def weighted_rating(x, m=m, C=C):

    v = x['Reviews']

    R = x['User Rating']

    return (v/(v+m) * R) + (m/(m+v) * C)
df['Weighted Rating'] = df.apply(weighted_rating, axis=1)
df.groupby(['Name','Author','Genre'], as_index=False)[['User Rating', 'Reviews', 'Weighted Rating']].mean().sort_values(by='Weighted Rating', ascending=False).head(10)
df.groupby(['Genre'])['Weighted Rating'].mean()
df.groupby(['Name', 'Author', 'Genre'], as_index=False)['Price'].mean().sort_values('Price', ascending=False).head(10)
df.groupby(['Genre'], as_index=False)['Price'].mean()
# Most Reviewed

df.groupby(['Author','Name', 'Genre'],as_index=False)[['Weighted Rating', 'Reviews']].mean().sort_values('Reviews', ascending=False).head(10)
df.groupby(['Genre'])['Reviews'].mean()