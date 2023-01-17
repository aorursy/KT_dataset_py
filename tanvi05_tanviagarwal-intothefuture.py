# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import operator
data_train = pd.read_csv('/kaggle/input/into-the-future/train.csv')

data_test = pd.read_csv('/kaggle/input/into-the-future/test.csv')
data_train.head()
data_train.describe()
data_train.info()
train = data_train.copy()
data_test.head()
data_test.describe()
data_test.info()
test = data_test.copy()
data_train['feature_1'].plot(kind = 'line', color = 'green', figsize = (5,5))
data_train['feature_2'].plot(kind = 'line', color = 'blue', figsize = (5,5))
pd.isna(data_train).sum()
pd.isna(data_test).sum()
data_train['time'] = pd.to_datetime(data_train['time'])

data_train.head()
data_train.shape
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

data_train[['feature_1', 'feature_2']] = scaler.fit_transform(data_train[['feature_1', 'feature_2']])
data_train
data_train.index = data_train['time']

data_train
data_train = data_train.drop(['id', 'time'], axis = 1)
data_train.head()
corr = data_train.corr()

sns.heatmap(corr, vmax=-1.0, vmin=-0.6)
feature_1_inspect = train['feature_1']

feature_1_inspect = feature_1_inspect.append(pd.Series(test['feature_1'].values))
len(feature_1_inspect.values)
feature_1_inspect.describe()
plt.plot(feature_1_inspect.values)
plt.plot(data_train['feature_1'])
plt.plot(data_test['feature_1'])
plt.plot(train.loc[30:180, 'feature_1'])
plt.plot(train.loc[30:180, 'feature_2'])
sns.heatmap(train.corr())
sns.pairplot(data=data_train)
train_copy = train.copy()
train = train.drop(['id' , 'time'], axis = 1)

train.head()
test_copy = test.copy()
test = test.drop(['time'], axis = 1)

test.head()
test[['feature_1', 'id']] = scaler.transform(test[['feature_1', 'id']])

test.head()

test['feature_1'].plot()
scaler = StandardScaler()

train[['feature_1', 'feature_2']] = scaler.fit_transform(train[['feature_1', 'feature_2']])

train.head()
sns.relplot(x='feature_1', y='feature_2', data=train, kind='scatter')
train['feature_1'] = train[train['feature_1'] < 3]

train = train.dropna()

train.describe()
sns.regplot(x='feature_1', y='feature_2', data=train)
from sklearn.model_selection import train_test_split, cross_val_score

x_train,x_test, y_train,  y_test = train_test_split(train['feature_1'], train['feature_2'], test_size=0.2)
x_train, y_train = pd.DataFrame(x_train, columns=['feature_1']), pd.DataFrame(y_train, columns=['feature_2'])

x_test, y_test = pd.DataFrame(x_test, columns=['feature_1']), pd.DataFrame(y_test, columns=['feature_2'])
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(x_train, y_train)
pred_test = LR.predict(x_test)

print(pred_test)
r2_score(y_test, pred_test)
plt.figure(figsize=(5, 5))

plt.scatter(x_train, y_train, color = "green")

plt.plot(x_train, LR.predict(x_train), 'go')

plt.title("LR Fit")

plt.xlabel("feature_1")

plt.ylabel("feature_2")

plt.show()

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()

knn_param = {'n_neighbors': [1, 3, 5],

            'leaf_size': [10, 20, 40, 50]}
grid_search = GridSearchCV(knn, knn_param, n_jobs=-1, cv=10)



grid_search.fit(x_train, y_train)
model = grid_search.best_estimator_

model.fit(x_train, y_train)
pred = model.predict(x_test)
r2_score(y_test, pred )
plt.figure(figsize = (5,5))

plt.scatter(x_train, y_train, color = "red")

plt.plot(x_train, model.predict(x_train), 'go')

plt.title("Knn Fit")

plt.xlabel("feature_1")

plt.ylabel("feature_2")

plt.show()
x_test = pd.DataFrame(test['feature_1'], columns=['feature_1'])

pred_test = pd.DataFrame(model.predict(x_test), columns=['feature_2'])

pred_test
pred_test_data = pd.concat([test_copy['id'], pred_test], axis=1, copy=False)

pred_test_data
id_copy = test_copy['id']
pred_test_data[['id', 'feature_2']] = scaler.inverse_transform(pred_test_data[['id', 'feature_2']])

pred_test_data['id'] = id_copy

print(pred_test_data.shape)

print(pred_test_data.head())
len(id_copy)
pred_test_data['feature_2']
submission = pd.DataFrame({'id': id_copy, 'feature_2':pred_test_data['feature_2'].values})
submission.head()
submission.to_csv(r'submission.csv', index=False)