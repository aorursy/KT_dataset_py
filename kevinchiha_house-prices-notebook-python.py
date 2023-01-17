# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
s = df.isnull().sum()

print(s[s!=0])
list = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 

        'BsmtExposure' ,'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',

        'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',

        'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']



for i in range(len(list)):

    df.loc[:, list[i]].fillna('None', inplace=True)

    df_test.loc[:, list[i]].fillna('None', inplace=True)

    

s = df.isnull().sum()

print(s[s!=0])

s2 = df_test.isnull().sum()

print(s2[s2!=0])
df.loc[:, 'LotFrontage'].fillna(df.loc[:, 'LotFrontage'].median(), inplace=True)

df_test.loc[:, 'LotFrontage'].fillna(df_test.loc[:, 'LotFrontage'].median(), inplace=True)
index = df[pd.isnull(df.loc[:, 'Electrical'])].Electrical.index.values

df = df.set_value(index=index, col='Electrical', value='SBrkr')
df.loc[:, df.isnull().sum() != 0]

for i in range(len(df_test.columns)):

    if df_test.isnull().sum()[i] != 0:

        df_test[df_test.isnull().sum().index[i]].fillna(df_test[df_test.isnull().sum().index[i]].value_counts().max(), inplace=True)

df_obj = df.loc[:, df.dtypes == 'object']

print(len(df_obj.columns))

df_obj = pd.get_dummies(df_obj)

df_obj.head()
df_obj_test = df_test.loc[:, df_test.dtypes == 'object']

print(len(df_obj_test.columns))

df_obj_test = pd.get_dummies(df_obj_test)

df_obj_test.head()
df_num = df.loc[:, df.dtypes != 'object']

df_num.tail()
df_num_test = df_test.loc[:, df_test.dtypes != 'object']

df_num_test.head()
df_new = pd.concat([df_obj, df_num], axis=1, join='outer')

df_new.tail()

df_new.columns
df_new_test = pd.concat([df_obj_test, df_num_test], axis=1, join='outer')

df_new_test.tail()

df_new_test.columns
df_new.dropna(axis=0, inplace=True)

y = df_new.SalePrice

df_new.drop('SalePrice', axis=1, inplace=True)
#from sklearn.model_selection import train_test_split

#X_train, X_test, label_train, label_test = train_test_split(df_new, y, test_size=0.15, random_state=7)
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt



    

pca = PCA(n_components=35)

pca.fit(df_new) #(X_train)



pcat = PCA(n_components=35)

pcat.fit(df_new_test) #(X_train)



df_new_fitted = pca.transform(df_new) #(X_train)

df_new_test_fitted = pcat.transform(df_new_test)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(df_new_fitted, y) #(data_train, label_train)

#lin_reg.score(data_test, label_test)



predictions = lin_reg.predict(df_new_test_fitted)

pd.DataFrame(predictions)



submission = pd.DataFrame({

        "Id": df_new_test["Id"],

        "SalePrice": predictions

    })

submission.to_csv('C:/Users/antchiha/Desktop/houses.csv', index=False)