# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os


# Any results you write to the current directory are saved as output.
df_test = pd.read_csv('../input/test.csv')
#RenanLevenhagen
df_train = pd.read_csv('../input/train.csv')
df_train.head()
#RenanLevenhagen
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
print(df_train.isnull().sum().max())
df_test = df_test.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_test = df_test.drop(df_test.loc[df_test['Electrical'].isnull()].index)
print(df_test.isnull().sum().max())
df_test.isnull().sum().sort_values(ascending=False)
var_to_fill = list(df_test.select_dtypes(exclude=object).isnull().sum().sort_values(ascending=False)[0:8].index)
df_test[var_to_fill]=df_test[var_to_fill].fillna(0)
df_test.select_dtypes(exclude=object).isnull().sum().sort_values(ascending=False)
df_train.shape
df_test.shape
var_categ = list(df_train.select_dtypes(include=object).columns)
var_categ
df_train.drop(labels=var_categ, axis=1,inplace=True)
df_train.columns
df_test.drop(labels=var_categ, axis=1,inplace=True)
df_test.columns
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df_train.corr(), vmax=1,vmin=-1, square=True)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df_train[['GarageArea','GarageCars','SalePrice']].corr(), vmax=1,vmin=-1, square=True, annot=True)

df_train.drop(labels='GarageArea', axis=1, inplace=True)
df_test.drop(labels='GarageArea', axis=1, inplace=True)

corrmat = df_train.corr()
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
cm

sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df_train.drop(labels='Id', axis=1, inplace=True)
test_id=df_test.pop('Id')
X = df_train.drop(labels='SalePrice',axis=1)
y = df_train.SalePrice
lm = LinearRegression()
lm.fit(X,y)
predict = lm.predict(df_test)
predict
df_final= pd.DataFrame(data={'Id':test_id, 'SalePrice':predict})
df_final.head()
df_final.to_csv('House_Price_Prediction',sep=',', index=False)
