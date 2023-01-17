# import liberies

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import random

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder

import matplotlib as mpl



mpl.style.use('ggplot')

%matplotlib inline
data = pd.read_csv('../input/cereal.csv')
# first 10 rows

print(data.head(10))
# Drop the first index

data.drop(data.index[0], inplace=True)
# Check to see if gone

print(data.head(1))
# check if theres any null values

print(data.isnull().sum())
# check data types

print(data.info())

# in order to continue to the visualization of our data we need

#to change the object types to numarical and categorical values
print(data.describe())
print(data.columns)
print(data.values)
# change to numberic

num_chnge = ['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups', 'rating']

data[num_chnge] = data[num_chnge].apply(pd.to_numeric)
# Check types

print(data.info())
# ok great now change mfr & type to categorical

categ = ['type','mfr']

for col in categ:

    data[col] = data[col].astype('category')
# Check

print(data[categ].info())
print(data.head())
tidy_set = pd.melt(frame=data, id_vars=['mfr', 'rating','name'], value_vars=['type', 'calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars','potass'], value_name='value')
# Sorted by values

print(tidy_set.sort_values('rating', ascending=False).head(10))

print('\n')

print('_____________________________________________________________')

print('\n')

print(tidy_set.sort_values('rating', ascending=True).head(10))
plt.figure(figsize=(10,10))

plt.title('Brand rating')

sns.barplot(x='mfr', y='rating', hue='shelf',data=data)
sns.factorplot(x='mfr', y='sugars',hue='shelf',col='shelf', data=data)
sns.jointplot(x='rating', y='sugars',data=data)
print(data.columns)
sns.distplot(data.sodium)