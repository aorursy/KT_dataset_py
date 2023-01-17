# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

nfx = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')

nfx.head(4)
nfx.info()
print('Number of distinct countries : {0}'.format(len(nfx.country.unique())))
plt.figure(figsize=(20,10))

sns.countplot(x='country', data=nfx, order=nfx.country.value_counts().index[0:20])
nfx[nfx.country == 'United States'].shape
plt.figure(figsize=(20,10))

sns.countplot(x='release_year', data=nfx[nfx.country=='United States'], order=nfx.release_year.value_counts().index[0:10])
"""Show type by countries"""

plt.figure(figsize=(20,10))

sns.countplot(x='country',hue='type',data=nfx[nfx.country.isin(['United States','India','United Kingdom','Japan','Canada'])])