import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

%matplotlib inline
train = pd.read_csv('../input/akvelon1/train.csv')

test = pd.read_csv('../input/akvelon1/test.csv')
train['Gender'] = train['Gender'].map({'male': 0, 'female': 1})
alco_mean = train.groupby('Alcohol')['Gender'].mean()

train['Alcohol'] = train['Alcohol'].map(alco_mean)

test['Alcohol'] = test['Alcohol'].map(alco_mean)
train['Alcohol'].unique()
train['Alcohol'] = train['Alcohol'].fillna(train['Gender'].mean())

test['Alcohol'] = test['Alcohol'].fillna(train['Gender'].mean())
train['Alcohol'].unique()
sns.countplot(train['Gender'])
corr = train.corr(method='kendall')
corr['Gender'].sort_values(ascending=False)
cols_negative_corr = corr[corr['Gender'] < 0]['Gender'].index
for col in cols_negative_corr:

    if train[col].dtype == np.float64:

        train[col] = -1 * train[col].values
for col in cols_negative_corr:

    if test[col].dtype == np.float64:

        test[col] = -1 * test[col].values
corr = train.corr(method='kendall')
cat_cols = train.select_dtypes(include=[np.object]).columns
cat_cols
train['Alcohol'].unique()
for col in train.drop('Gender', axis=1).columns:

    if train[col].isnull().any() or test[col].isnull().any():

        if train[col].dtype == np.float64 or train[col].dtype == np.int64:

            mode = train[col].std()

            train[col] = train[col].fillna(mode)

            test[col] = test[col].fillna(mode)

        else:

            mode = train[col].value_counts().index[0]

            train[col] = train[col].fillna(mode)

            test[col] = test[col].fillna(mode)
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
for col in train.drop('Gender', axis=1).select_dtypes([np.object]).columns:

    encoder = LabelEncoder()

    train[col] = encoder.fit_transform(train[col])

    test[col] = encoder.transform(test[col])
pipe = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier())])
features_to_drop = list(corr[corr['Gender'] < 0.0].index)
cross_val_score(pipe, 

                train.drop(features_to_drop + ['Gender'], axis=1), 

                train['Gender'], cv=5)
score = []

for i in range(1, 100):

    pipe = Pipeline([('sc', StandardScaler()),

                     ('knn', KNeighborsClassifier(n_neighbors=i))])

    cv = cross_val_score(pipe, 

                train.drop(features_to_drop + ['Gender'], axis=1), 

                train['Gender'], cv=5)

    score.append(cv.mean())
max(score)
#оптимальное количество ближайших соседей

n_neighbors = np.argmin(score) + 1
n_neighbors
# пайплайн с оптимальным количеством соседей

pipe = Pipeline([('sc', StandardScaler()),('knn', KNeighborsClassifier(n_neighbors=n_neighbors))])
# переучиваем пайплайн на всех данных

pipe.fit(train.drop(features_to_drop + ['Gender'], axis=1), train['Gender'])
# получаем предсказание для теста

pred = pipe.predict(test.drop(features_to_drop, axis=1))
test['Gender'] = pred
# не забудьте перевернуть ваши коды обратно в строку

test['Gender'] = test['Gender'].map({0: 'male', 1: 'female'})
# создаем файл с предсказанием для теста

test[['Id', 'Gender']].to_csv('sub.csv', index=None)