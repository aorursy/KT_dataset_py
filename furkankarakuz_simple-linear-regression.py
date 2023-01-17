# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("../input/random-linear-regression/train.csv")

test=pd.read_csv("../input/random-linear-regression/test.csv")
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(15,8))

sns.lineplot(x=test.index,y=test["x"],ax=ax[0][0],color="#8E44AD")

sns.lineplot(x=test.index,y=test["y"],ax=ax[0][1],color="#27AE60")

sns.lineplot(x=train.index,y=train["x"],ax=ax[1][0],color="#8E44AD")

sns.lineplot(x=train.index,y=train["y"],ax=ax[1][1],color="#27AE60")
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(15,8))

sns.boxplot(test["x"],ax=ax[0][0],color="#8E44AD")

sns.boxplot(test["y"],ax=ax[0][1],color="#27AE60")

sns.boxplot(train["x"],ax=ax[1][0],color="#8E44AD")

sns.boxplot(train["y"],ax=ax[1][1],color="#27AE60")
train=train[train["x"]<3500]
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(15,8))

sns.boxplot(test["x"],ax=ax[0][0],color="#8E44AD")

sns.boxplot(test["y"],ax=ax[0][1],color="#27AE60")

sns.boxplot(train["x"],ax=ax[1][0],color="#8E44AD")

sns.boxplot(train["y"],ax=ax[1][1],color="#27AE60")
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error,mean_absolute_error
X_train,X_test,Y_train,Y_test=train[["x"]],test[["x"]],train["y"],test["y"]

X_train=sm.add_constant(X_train)

X_test=sm.add_constant(X_test)
stats_model=sm.OLS(Y_train,X_train).fit()

print(stats_model.summary())
plt.figure(figsize=(16,5))

plt.plot(stats_model.resid)
stats_model.params
print("Y =",round(stats_model.params[0],6),"+ X *",round(stats_model.params[1],6))
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,5))

sns.regplot(x=X_test["x"],y=Y_test,scatter_kws={"lw":0.1,"color":"#8E44AD"},line_kws={"lw":3,"color":"red"},ax=ax[0])

sns.regplot(x=X_train["x"],y=Y_train,scatter_kws={"lw":0.1,"color":"#27AE60"},line_kws={"lw":3,"color":"red"},ax=ax[1])

ax[0].set_title("Y = -0.107265 + X * 1.000656")

ax[1].set_title("Y = -0.107265 + X * 1.000656")
mae=mean_absolute_error(Y_train,stats_model.predict(X_train))

mse=mean_squared_error(Y_train,stats_model.predict(X_train))

rmse=np.sqrt(mse)

print("Train Mean Absolute Error (MAE) : ",mae)

print("Train Mean Squared Error  (MSE) : ",mse)

print("Train Root Mean Squared Error  (RMSE) : ",rmse)
mae=mean_absolute_error(Y_test,stats_model.predict(X_test))

mse=mean_squared_error(Y_test,stats_model.predict(X_test))

rmse=np.sqrt(mse)

print("Test Mean Absolute Error (MAE) : ",mae)

print("Test Mean Squared Error  (MSE) : ",mse)

print("Test Root Mean Squared Error  (RMSE) : ",rmse)