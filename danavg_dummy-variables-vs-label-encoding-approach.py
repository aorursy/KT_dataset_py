import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns

from sklearn import preprocessing
train = pd.read_csv('../input/train.tsv', sep='\t')

test = pd.read_csv('../input/test.tsv', sep='\t')
train.head(5)
train.dtypes
test.head(5)
test.dtypes
# checking for Nulls

obj = train.select_dtypes(include=['object']).copy()

train[obj.isnull().any(axis=1)].head(5)
train["category_name"].value_counts().head()
train["brand_name"].value_counts().head()
df_dummies = pd.get_dummies(train['category_name'])

df_dummies.head()
df_new = pd.concat([train['price'], df_dummies], axis=1)

df_new.head()
train["category_name"].value_counts().head()
train["brand_name"].value_counts().head()
encoder = preprocessing.LabelEncoder()

train["brand_name"] = encoder.fit_transform(train["brand_name"].fillna('Nan'))

train["category_name"] = encoder.fit_transform(train["category_name"].fillna('Nan'))

train.head()