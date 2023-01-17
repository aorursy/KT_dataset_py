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
df = pd.read_csv('../input/scotch_review.csv')
df.head()
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()
df.dtypes
df.shape
df.category.value_counts()
print(df.price.max())
print(df.price.min())
set(df.currency)
df['review.point'].value_counts()[:10]
df[['category','review.point','price']].sort_values('review.point',ascending=False)[:10]
df[('category')].value_counts().plot(kind='bar')