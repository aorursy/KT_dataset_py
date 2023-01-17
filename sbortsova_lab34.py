# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from os.path import join as pjoin

import os

import warnings

import numpy as np

from scipy  import sparse

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

#графики в svg выглядят более четкими

%config InlineBackend.figure_format = 'svg' 



#увеличим дефолтный размер графиков

from pylab import rcParams

rcParams['figure.figsize'] = 8, 5

import pandas as pd

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

os.getcwd()

os.listdir("../input/")
PATH_TO_DATA = '../input/lab34classificationtable/'
df = pd.read_csv(pjoin(PATH_TO_DATA, 'train.csv'))

df.head()
df_test = pd.read_csv(pjoin(PATH_TO_DATA, 'test.csv'))

df_test.head()
df.info()
df['y'].value_counts()
df['y'].value_counts().plot(kind='bar', label='y')

plt.legend()

plt.title('Распределение отказа');
df['contact_date'].astype('datetime64[ns]')

df['birth_date'].astype('datetime64[ns]')
df_test['contact_date'].astype('datetime64[ns]')

df_test['birth_date'].astype('datetime64[ns]')
df ['age'] = (pd.to_datetime(df['contact_date'])- pd.to_datetime(df['birth_date']))

df['age'].astype('int64')

df['pdays'].astype('int64')

print(df['age'])
df_test ['age'] = (pd.to_datetime(df['contact_date'])- pd.to_datetime(df['birth_date']))

df_test['age'].astype('int64')

df_test['pdays'].astype('int64')
categorical_columns = [c for c in df.columns if df[c].dtype.name == 'object']

numerical_columns   = [c for c in df.columns if df[c].dtype.name != 'object']

numerical_columns.remove('Unnamed: 0')

print(categorical_columns)

print(numerical_columns)
df[categorical_columns].describe()
for c in categorical_columns:

    print(c, df[c].unique())
data_describe = df.describe(include=[object])

binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]

nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]

print("Бинарные", binary_columns)

print("Небинарные", nonbinary_columns)
df.at[df['contact'] == 'cellular', 'contact'] = 0

df.at[df['contact'] == 'telephone', 'contact'] = 1

df['contact'].describe()
df_test.at[df_test['contact'] == 'cellular', 'contact'] = 0

df_test.at[df_test['contact'] == 'telephone', 'contact'] = 1

df_test['contact'].describe()
nonbinary_columns.remove('contact_date')

nonbinary_columns.remove('birth_date')

nonbinary_columns.remove('default')

numerical_columns.remove('y')
data_nonbinary = pd.get_dummies(df[nonbinary_columns])

print(nonbinary_columns)

print(data_nonbinary.columns)
data_nonbinary_test = pd.get_dummies(df_test[nonbinary_columns])
data_numerical = df[numerical_columns]

data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()

data_numerical.describe()
data_numerical_test = df_test[numerical_columns]

data_numerical_test = (data_numerical_test - data_numerical_test.mean()) / data_numerical_test.std()

data_numerical_test.describe()
data_train = pd.concat((data_numerical, df[binary_columns], data_nonbinary), axis=1)

data_train = pd.DataFrame(data_train, dtype=float)

print(data_train.shape)

print(data_train.columns)
data_test = pd.concat((data_numerical_test, df_test[binary_columns], data_nonbinary_test), axis=1)

data_test = pd.DataFrame(data_test, dtype=float)

print(data_train.shape)

print(data_train.columns)
y_train = df['y']

X_train=data_train

feature_names = X_train.columns

print('features_name', feature_names)
X_test=data_test

feature_names = X_test.columns

print('features_name', feature_names)
X_train_drop, X_test_drop, y_train_drop, y_test_drop = train_test_split(X_train, y_train, test_size = 0.3, random_state = 11)



N_train, _ = X_train_drop.shape 

N_test,  _ = X_test_drop.shape 

print(N_train, N_test)
n_neighbors_array = [1, 3, 5, 7, 10, 15]

knn = KNeighborsClassifier()

grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})

grid.fit(X_train_drop, y_train_drop)



best_cv_err = 1 - grid.best_score_

best_n_neighbors = grid.best_estimator_.n_neighbors

print('Лучшая ошибка', best_cv_err, 'для лучшего количества соседей', best_n_neighbors)
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)

knn.fit(X_train_drop, y_train_drop)



err_train = np.mean(y_train_drop != knn.predict(X_train_drop))

#test_y = knn.predict(X_test)

err_test  = np.mean(y_test_drop  != knn.predict(X_test_drop))

print('Ошибка тренировочных данных', err_train)

print('Ошибка тестовых данных', err_test)

#print('Test_y', test_y)
from sklearn import ensemble

rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)

rf.fit(X_train_drop, y_train_drop)



err_train = np.mean(y_train_drop != rf.predict(X_train_drop))

err_test  = np.mean(y_test_drop  != rf.predict(X_test_drop))

print('Ошибка тренировочных данных', err_train)

print('Ошибка тестовых данных', err_test)
importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]



print("Feature importances:")

for f, idx in enumerate(indices):

    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
d_first = 20

plt.figure(figsize=(8, 8))

plt.title("Feature importances")

plt.bar(range(d_first), importances[indices[:d_first]], align='center')

plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)

plt.xlim([-1, d_first]);
best_features = indices[:12]

best_features_names = feature_names[best_features]

print(best_features_names)
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)

gbt.fit(X_train_drop[best_features_names], y_train_drop)



err_train = np.mean(y_train_drop != gbt.predict(X_train_drop[best_features_names]))

err_test = np.mean(y_test_drop != gbt.predict(X_test_drop[best_features_names]))

print('Ошибка тренировочных данных', err_train)

print('Ошибка тестовых данных', err_test)
gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)

gbt.fit(X_train[best_features_names], y_train)



res = gbt.predict_proba(X_test[best_features_names])[:, 1]

result = pd.DataFrame({'y': res, 'id': df_test.index})

print('result Y', res, result)

result.to_csv('res_baseline.csv', index=False)