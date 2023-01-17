# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#dataread

dataset = pd.read_csv('../input/data.csv',encoding = "ISO-8859-1")

dataset.describe()
dataset.head()
df = dataset.drop(["location_monitoring_station","location","agency","sampling_date","rspm","spm","pm2_5"], axis = 1)

df = df.dropna()

# here we are replacing names for one of the categories and combining them into one name i.e. Residential Area

df = df.replace(to_replace = 'Residential, Rural and other Areas', value = 'Residential Area')

df.head()
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year

df.head(10)
sns.pairplot(df[['so2','no2']])