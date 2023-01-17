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
df = pd.read_csv('../input/GrammarandProductReviews.csv')
df.head(3)
df.shape
df.columns
df['reviews.rating'].unique()
df['reviews.title'].unique()[:5]
df['reviews.rating'].value_counts().sort_values(ascending=False)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.countplot(df['reviews.rating'])
plt.show()
df['reviews.didPurchase'].value_counts().sort_values(ascending=False)
df['reviews.didPurchase'].isnull().sum()
print('total rows: ',df.shape[0])
print('missing reviews: 38886' )
print('total fake reviews : 28476'  )
print('genuine reviews: 3682')
sns.countplot(df['reviews.didPurchase'])
plt.show()
len(df['brand'].unique())
df['brand'].value_counts().sort_values(ascending=False)[:10]
df.groupby('brand')['reviews.rating'].agg(['mean','count']).sort_values(by='count',ascending=False)[:10]
len(df['categories'].unique())
df['categories'].value_counts().sort_values(ascending=False)[:5]
df['categories'][:10]
df[df['reviews.rating']==5].head()
brands_ratings = df.groupby(['brand'])
brands_ratings.head()
df['reviews.text'][0]
