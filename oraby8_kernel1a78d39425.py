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
data=pd.read_csv('../input/jamalon-arabic-books-dataset/jamalon dataset.csv')

data.describe()
plt.figure(figsize=(20,10))

data[['Category','Price']].groupby('Category').mean().plot(kind='bar')
data['Publication year'].value_counts()
data=data[data['Publication year']>1999]
g=sns.FacetGrid(data,col='Category')

g.map(plt.hist,'Publication year',bins=20)