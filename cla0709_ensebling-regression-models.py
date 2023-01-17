# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_columns = df.select_dtypes(include=numerics).columns
categorical_columns = df.select_dtypes(exclude=numerics).columns
numerics_df = df.select_dtypes(include=numerics).fillna(0)
categorical_df = df.select_dtypes(exclude=numerics)

categorical_df = categorical_df.applymap(lambda a: str(a))
categorical_df = categorical_df.apply(preprocessing.LabelEncoder().fit_transform)

df[numeric_columns] = numerics_df
df[categorical_columns] = categorical_df

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
n = len(X)
d = len(X.T)
K = 10

# K-Fold
indices = np.random.choice(n,n, replace = False)
X_k = np.zeros((K,int(n/K),d))
y_k = np.zeros((K,int(n/K)))
for k in range(0,K):
    index_k = np.array(indices[int(k*(n/K)):int((k+1)*(n/K))])
    X_k[k] = X[index_k]
    y_k[k] = y[index_k]

# PCR
# HyperParams:
# 1- n_components

# Error for each d, k times
error_k_d = np.zeros((K,int(d-2)))

for k in range (0,K):
    X_test = X_k[k]
    y_test = y_k[k]
    X_train = np.delete(X_k.copy(),k,axis=0).reshape(int((K-1)*(n/K)),d)
    y_train = np.delete(y_k.copy(),k,axis=0).reshape(int((K-1)*(n/K)),)
    for i in range(2,d):
        X_PCA = PCA(n_components=i).fit(X_train)
        X_PCA_train = X_PCA.transform(X_train)
        linear_model = LinearRegression().fit(X_PCA_train,y_train)
        y_pred = linear_model.predict(X_PCA.transform(X_test))
        mse = mean_squared_error(y_test, y_pred)
        msle = np.sum(np.log((y_pred + 1)/(y_test + 1)))
        error_k_d[k][i-2] = msle
plt.figure(figsize=(10,10))
plt.scatter(np.arange(0,d-4),error_k_d.mean(axis=0)[:-2], alpha=0.6)
plt.show()
# 43
print(error_k_d.mean(axis=0)[43])
# Random Forest Regression
max_depth = int(np.log2(0.9*n))
trees = 8
error_k_d = np.zeros((K,trees-1,max_depth-1))
for k in range (0,K):
    X_test = X_k[k]
    y_test = y_k[k]
    X_train = np.delete(X_k.copy(),k,axis=0).reshape(int((K-1)*(n/K)),d)
    y_train = np.delete(y_k.copy(),k,axis=0).reshape(int((K-1)*(n/K)),)
    for log_t in range(1,trees):
        t = np.power(2,log_t) 
        for md in range(1,max_depth):
            RFR = RandomForestRegressor(n_estimators=t, max_depth=md, max_features='sqrt')
            RFR.fit(X_train, y_train)
            y_pred = RFR.predict(X_test)
            msle = np.sum(np.log((y_pred + 1)/(y_test + 1)))
            error_k_d[k][log_t-1][md-1] = msle

plt.figure(figsize=(10,10))
cose=error_k_d.mean(axis=0)
print(cose)
plt.scatter(cose[:,0],cose[:,1], alpha=0.6)
plt.show()
