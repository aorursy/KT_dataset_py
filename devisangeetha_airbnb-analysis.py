# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
air_data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

air_data.info()
air_data.head(5)
sns.distplot(air_data['number_of_reviews'],kde=False,bins=10)
fig, ax = plt.subplots()

sns.distplot(air_data['number_of_reviews'],kde=False,bins=30)

ax.set_xlim(0,300)
air_data['room_type'].value_counts()
sns.countplot(x='room_type',data=air_data)
sns.countplot(x='neighbourhood_group',data=air_data)
avg_price=air_data.groupby('neighbourhood_group').agg({'price':'mean'}).sort_values(by='price',ascending=False)

avg_price.reset_index()

sns.barplot(x=avg_price.index,y='price',data=avg_price)
sns.boxplot(x='neighbourhood_group',y='price',data=air_data)
mprice=air_data[air_data['price']<=1000]

sns.boxplot(x='neighbourhood_group',y='price',data=mprice)