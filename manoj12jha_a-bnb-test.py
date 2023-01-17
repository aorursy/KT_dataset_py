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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', delimiter=',')

df.head()
df.shape
df.dtypes
df.isnull().values.any()
df.isnull().sum()
#df.drop(['id','host_name','last_review'], axis=1, inplace=True)

df.drop(['id','host_name','last_review'], axis=1, inplace=True)

#df.isnull().sum()
df.isnull().sum()
df.head(10)
df.fillna({'reviews_per_month':0}, inplace=True)

df.isnull().sum()
df.neighbourhood_group.unique()
df.neighbourhood .unique()
df.room_type.unique()
df.head(8)
top_host = df.host_id.value_counts().head(10)

top_host
top_host_check=df.calculated_host_listings_count.max()

top_host_check
import matplotlib.pyplot as plt

import seaborn as sns
#plt.figure(figsize=(10,10))

sns.set(rc={'figure.figsize':(10,10)})

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=df)
df.dtypes