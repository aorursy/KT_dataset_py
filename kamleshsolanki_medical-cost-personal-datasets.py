# data load and helper library

import numpy as np

import pandas as pd

import os

# visulization and insight

import seaborn as sns

import matplotlib.pyplot as plt

# data preprocessing and encode

from sklearn.preprocessing import LabelEncoder, StandardScaler

# data modeling and prediction

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.svm import SVR
data = pd.read_csv('../input/insurance/insurance.csv')
data.head()
data.info()
data.isna().sum()
categorical_columns = ['sex', 'smoker', 'region']

numerical_columns = ['age', 'bmi', 'children', 'charges']
print('duplicated : {}'.format(data.duplicated().sum()))

data = data.drop_duplicates(keep = 'last')

print('duplicated after remove: {}'.format(data.duplicated().sum()))
target = 'charges'

for col in numerical_columns:

    sns.boxplot(data = data, x = col)

    plt.show()
data = data[data.charges < 50000]

sns.boxplot(data = data, x = 'charges')
data[categorical_columns].nunique().plot(kind = 'bar')
for col in categorical_columns:

    sns.catplot(x = col, data = data, kind = 'count')
for col in categorical_columns:

    data[col] = LabelEncoder().fit_transform(data[col])
data.describe()
for col in numerical_columns:

    sns.distplot(data[col], rug = True)

    plt.show()
sns.heatmap(data.corr())

plt.show()

data.corr()
sns.jointplot(x = 'age', y = 'charges', data = data)
sns.jointplot(x = 'smoker', y = 'charges', data = data)
sns.catplot(x = 'age', y = 'charges', data = data, hue = 'smoker')
for col in categorical_columns:

    data[col] = LabelEncoder().fit_transform(data[col])
for col in categorical_columns:

    data[col] = data[col].astype('category')

for col in numerical_columns:

    data[col] = pd.to_numeric(data[col])
train, charges = data, data['charges']

X_train, X_test, Y_train, Y_test = train_test_split(train, charges, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.ensemble import RandomForestRegressor
performance = {'algo' : [], 'r2_score' : []}
lr = LinearRegression().fit(X_train,Y_train)



y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



r2 = r2_score(y_test_pred, Y_test)

print('r2_score: {}'.format(r2))
performance['algo'].append('LinearRegression')

performance['r2_score'].append(r2)
lr = RandomForestRegressor().fit(X_train,Y_train)



y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



r2 = r2_score(y_test_pred, Y_test)

print('r2_score: {}'.format(r2))
performance['algo'].append('RandomForestRegressor')

performance['r2_score'].append(r2)
performance_df = pd.DataFrame(performance)
sns.plotting_context
sns.catplot(data = performance_df, x = 'algo', y = 'r2_score', kind = 'bar')