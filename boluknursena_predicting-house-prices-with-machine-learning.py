import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import warnings

pd.set_option('display.float_format', lambda x: '%.3f' % x)

sns.set(style='white', context='notebook', palette='deep')

warnings.filterwarnings('ignore')

sns.set_style('white')

%matplotlib inline



data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.head(10)
print(data_train.shape) # Train data's shape

print(data_train.columns)

print(data_test.shape) # Train data's shape

print(data_test.columns)
data_train['SalePrice'].describe() # Stistics summary
sns.distplot(data_train['SalePrice']); # histogram
data_train['SalePrice'].skew() # Skewness
data_train['SalePrice'].kurt() #kurtosis
data_null_train = data_train.isnull().sum().sort_values(ascending=False)

data_null_train.head(10)
from sklearn import preprocessing

for col in data_null_train.index:

    if data_train[col].dtypes == 'object':

         data_train[col] = data_train[col].fillna("None")

    else:

        data_train[col] = data_train[col].fillna(data_train[col].mean())

data_train.head(10)
data_null_test = data_test.isnull().sum().sort_values(ascending=False)



for col in data_null_test.index:

    if data_test[col].dtypes == 'object':

         data_test[col] = data_test[col].fillna("None")

    else:

        data_test[col] = data_test[col].fillna(data_test[col].mean())

data_test.head(10)
data_train.corr()
plt.subplots(figsize=(40, 40))

sns.heatmap(data_train.corr(), center=0, cmap="YlGnBu")
columns = data_train.corr().nlargest(10,'SalePrice')['SalePrice'].index

positiveCorrelation = np.corrcoef(data_train[columns].values.T)



plt.subplots(figsize=(10, 10))

sns.heatmap(positiveCorrelation, cbar=True, annot=True, yticklabels=columns.values, xticklabels=columns.values,cmap="YlGnBu")

data_all = pd.concat((data_train.iloc[:,1:-1],

                      data_test.iloc[:,1:]))
from scipy.stats import skew

numeric_feats = data_all.dtypes[data_all.dtypes != "object"].index

skewed_feats = data_all[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



data_all[skewed_feats] = np.log1p(data_all[skewed_feats])

data_all = pd.get_dummies(data_all)

data_train.SalePrice = np.log1p(data_train.SalePrice)

y = data_train.iloc[:, -1].values



X_train = data_all[:data_train.shape[0]]

X_test = data_all[data_train.shape[0]-1:]
print(X_train.shape)

print(X_test.shape)
from sklearn.tree import DecisionTreeRegressor



dt = DecisionTreeRegressor(random_state=1)

dt.fit(X_train, y)
dt_score = dt.score(X_train, y)

dt_predict = dt.predict(X_test)

print("DecisionTreeRegressor")

print("Score: ", dt_score)

print("Predict: ", dt_predict)
from sklearn.ensemble import GradientBoostingRegressor



gb = GradientBoostingRegressor()

gb.fit(X_train, y)
gb_score = gb.score(X_train, y)

gb_predict = gb.predict(X_test)

print("DecisionTreeRegressor")

print("Score: ", gb_score)

print("Predict: ", gb_predict)