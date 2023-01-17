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
# filepath = '/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv'

# read the data from csv

#df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df = pd.read_csv('/kaggle/input/suicide-data-with-continent/Suicide data with Continent.csv')

#df.describe()

df.info()
# drop 'HDI for year' column

#df = df.drop(columns=['HDI for year','country-year'])

df
df.describe()

#df.shape
df.isnull().sum()

len(df['country'].unique())
import matplotlib.pyplot as plt

import seaborn as sns



#cat_year = sns.catplot('sex','suicides_no', hue='age', col='year', data = df, kind='bar',col_wrap=3)

cat_year = sns.catplot('year','suicides_no', col='Continent', data = df, kind='bar',col_wrap=3)