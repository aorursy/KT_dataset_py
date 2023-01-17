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
#Libraries

from matplotlib import pyplot as plt

import pandas as pd



data_url = '../input/home-data-for-ml-course/train.csv'



df = pd.read_csv(data_url, index_col=0) #index_col=0 means first column will become index, otherwise specify with column name 'example name'





print (df.head())

print (df.dtypes)

print (df.shape)

print (df.columns)





df=df.select_dtypes(include=['number'])

df.describe()
corr = df.corr()

print(corr)



import statsmodels.api as sm

sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.show()
df_2 = df[['OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars','GarageArea','SalePrice']]

#print (df_2['SalePrice'])
df_2.isna().sum()

print (df_2.shape)

df_2.dropna(axis = 0,how = 'any' ,inplace = True)

df_2.isna().sum()

print (df_2.shape)


X = df_2[['OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars','GarageArea']].values

y = df_2['SalePrice'].values

#print (X)

#print (y.shape)

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

print (X)

scaling = MinMaxScaler().fit(X)

X = scaling.transform(X)

print (X)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5) 

kf.get_n_splits(X)

print(kf)

print (kf.split(X))



for train_index, test_index in kf.split(X):

 print('TRAIN:', train_index, 'TEST:', test_index)

 X_train, X_test = X[train_index], X[test_index]

 y_train, y_test = y[train_index], y[test_index]



from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

y_predict=reg.predict(X_test)

tabulate = pd.DataFrame({'predict':y_predict, 'actual':y_test})

print (tabulate)

print (reg.coef_)



temp2=tabulate.head(21)

temp2.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
from sklearn.model_selection import cross_val_score

scores = cross_val_score(reg, X_train, y_train)

print("Cross-validation scores: {}".format(scores))
print ('Model score =', reg.score(X_train, y_train))



from sklearn.metrics import mean_squared_error

from math import sqrt



print ('Mean of y_test =',y_test.mean())

print('MS error =', mean_squared_error(y_predict,y_test))

print('RMS error =', sqrt(mean_squared_error(y_predict,y_test)))
test_url = '../input/home-data-for-ml-course/test.csv'

df_test = pd.read_csv(test_url, index_col=0)

ind=df_test.index

#print (df_test.dtypes)

df_test=df_test.select_dtypes(include=['number'])

#print (df_test.dtypes)



#print(df_test.isna().sum())



median=df_test['LotFrontage'].median()

df_test['LotFrontage'].fillna(median, inplace=True)



median=df_test['MasVnrArea'].median()

df_test['MasVnrArea'].fillna(median, inplace=True)



median=df_test['BsmtFinSF1'].median()

df_test['BsmtFinSF1'].fillna(median, inplace=True)



median=df_test['BsmtFinSF2'].median()

df_test['BsmtFinSF2'].fillna(median, inplace=True)



median=df_test['BsmtUnfSF'].median()

df_test['BsmtUnfSF'].fillna(median, inplace=True)



median=df_test['TotalBsmtSF'].median()

df_test['TotalBsmtSF'].fillna(median, inplace=True)



median=df_test['BsmtFullBath'].median()

df_test['BsmtFullBath'].fillna(median, inplace=True)



median=df_test['BsmtHalfBath'].median()

df_test['BsmtHalfBath'].fillna(median, inplace=True)



median=df_test['GarageYrBlt'].median()

df_test['GarageYrBlt'].fillna(median, inplace=True)



median=df_test['GarageCars'].median()

df_test['GarageCars'].fillna(median, inplace=True)



median=df_test['GarageArea'].median()

df_test['GarageArea'].fillna(median, inplace=True)



#df_test.dropna(axis = 0, how ='any', inplace = True)

#df_test.isna().sum()

print (df_test.shape)

#print (df.shape)



df_2 = df_test[['OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars','GarageArea']]

X = df_2[['OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars','GarageArea']].values





#X=df_test.iloc[:,0:36].values

#y=df_test.iloc[:,36].values





X = scaling.transform(X)



y_predict=reg.predict(X)

df_save = pd.DataFrame({'Id':ind,'SalePrice':y_predict})

df_save.to_csv("predict.csv") 