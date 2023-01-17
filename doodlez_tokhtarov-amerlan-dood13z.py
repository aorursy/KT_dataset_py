import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as mse
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_train.info(verbose=True)
df_train.describe().T
df_test.info()
len(df_train.drop_duplicates()) 
df_train['SalePrice'].hist()
df_train.columns
for x in df_train.columns:
    if df_train[x].isna().sum():
        print(x, df_train[x].isna().sum())
for x in ['Alley', 'PoolQC', 'Fence', 'MiscFeature']:
    f = 'z'+x
    df_train[x] = df_train[x].fillna(f)
    df_test[x] = df_test[x].fillna(f)
for x in df_train.select_dtypes(object).columns:
    if df_train[x].isna().sum() < df_train[x].count()*0.1 and df_train[x].isna().sum() > 0:
        df_train.dropna(subset = [x], inplace = True)
len(df_train)
for x in df_test.select_dtypes(['float64','int64']).columns[1:]:
    if df_test[x].isna().sum():
        print (x, df_test[x].isna().sum())
for x in df_test.select_dtypes(['float64','int64']).columns[2:]:
    if df_test[x].isna().sum():
        plt.figure()
        plt.hist(df_train[x])
        plt.xlabel(x)
for x in df_train.select_dtypes(['float64','int64']).columns[1:-1]:
    if df_train[x].isna().sum():
        print (x, df_train[x].isna().sum())
df_train['MasVnrArea'].hist()
df_test['MasVnrArea'].hist()
df_train.drop('MasVnrArea', axis = 1, inplace= True)
df_test.drop('MasVnrArea', axis = 1, inplace=True)
df_train['LotFrontage'].hist()
df_train['LotFrontage'][df_train['LotFrontage']< 150].hist()
lotFront = df_train['LotFrontage'][df_train['LotFrontage'] < 150].mean()
lotFront
df_train['LotFrontage'].fillna(lotFront, inplace = True)
df_test['LotFrontage'].fillna(lotFront, inplace = True)
for x in df_train.select_dtypes(['int64','float64']).columns[1:-1]:
    if not np.dtype(df_train[x]) == np.dtype(df_test[x]):
        print(x, np.dtype(df_train[x]), np.dtype(df_test[x]))
for x in df_train.select_dtypes(['int64']).columns[1:-1]:
    if not np.dtype(df_train[x]) == np.dtype(df_test[x]):
        df_train[x] = df_train[x].astype(np.dtype(df_test[x]))
for x in df_test.columns[1:]:
    if np.dtype(df_train[x]) != np.dtype(df_test[x]):
        print(x)
for x in df_train.select_dtypes(['float64','int64']).columns[1:-1]:
    if df_test[x].isna().sum():
        df_test[x].fillna(df_train[x].mean(), inplace = True)
mnmx = MinMaxScaler()
for x in df_test.drop(['Id'], axis = 1).select_dtypes(['float64', 'int64']).columns[1:]:
    mnmx.fit(df_train[x].values.reshape(-1,1))
    df_train[x] = mnmx.transform(df_train[x].values.reshape(-1,1))
    df_test[x] = mnmx.transform(df_test[x].values.reshape(-1,1))
df_train['FireplaceQu'].fillna('0', inplace = True)
df_train = pd.get_dummies(df_train, columns = ['CentralAir'], prefix='CA')
df_test = pd.get_dummies(df_test, columns = ['CentralAir'], prefix='CA')
for x in df_test.columns:
    if df_test[x].isna().sum():
        df_test[x].fillna(df_test[x].mode()[0], inplace = True)
        plt.figure()
        plt.hist(df_train[x])
        plt.xlabel(x + str(df_test[x].isna().sum()))
df_train.drop('SalePrice', axis =1).columns == df_test.columns
for x in df_train.select_dtypes(object).columns:
    label = LabelEncoder()  
    temp = []
    for i in df_train[x].unique():
        temp.append(i)
    for j in df_test[x].unique():
        temp.append(j)
    temp = list(set(temp))
    label.fit(temp)
    df_train[x] = label.transform(df_train[x])
    df_test[x] = label.transform(df_test[x])
x = df_train.drop(['Id', 'SalePrice'], axis = 1)

y = df_train['SalePrice']

testx = df_test.drop('Id',axis = 1)
results = {}
ith = np.arange(0.05, 0.7, 0.05)
jth = np.arange(0, len(df_train), len(df_train)//len(ith) +1)
for i in ith:
    for j in jth:
        mean_res = []
        for k in range(10):
            model = LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = i, random_state = j)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mean_res.append((mse(y_test, y_pred)))
        results[np.mean(mean_res)] = [i, j]
min(sorted(results))
results[min(sorted(results))]
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.65, random_state = 618)
model.fit(X_train, y_train)
y_pred = model.predict(testx)
df = pd.DataFrame()
df['Id'] = df_test['Id']
df['SalePrice'] = y_pred
df.to_csv('answer618.csv',index = False)
