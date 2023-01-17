# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from sklearn.linear_model import LinearRegression
train_data = pd.read_csv('../input/train.csv') #訓練用データ

test_data = pd.read_csv('../input/test.csv') #テスト用データ
test_data.head()
train_data2 = train_data[[

'MSSubClass',

'YearBuilt',

'YearRemodAdd',

'BsmtUnfSF',

'CentralAir',

'1stFlrSF',

'2ndFlrSF',

'TotRmsAbvGrd',

'SalePrice']]

train_data2.head()
test_data2 = test_data[[

'MSSubClass',

'YearBuilt',

'YearRemodAdd',

'BsmtUnfSF',

'CentralAir',

'1stFlrSF',

'2ndFlrSF',

'TotRmsAbvGrd']]

test_data2.head()
train_data2.isnull().any()
test_data2.isnull().any()
test_data2.fillna({'BsmtUnfSF':0},inplace=True) 
test_data2.isnull().any()
pd.get_dummies(train_data2['CentralAir']).head() 

pd.get_dummies(test_data2['CentralAir']).head() 
train_data2 = train_data2.join(pd.get_dummies(train_data2['CentralAir'],drop_first=True)) 

test_data2 = test_data2.join(pd.get_dummies(test_data2['CentralAir'],drop_first=True)) 
train_data2.head()
test_data2.head()
train_data2.drop(['CentralAir'], axis=1, inplace=True) 

test_data2.drop(['CentralAir'], axis=1, inplace=True) 
test_data2 = test_data2.rename(columns={'Y':'CentralAir'})
train_data2 = train_data2.rename(columns={'Y':'CentralAir'})
test_data2.head()
train_data2.head()
train_data2_y = train_data2[['SalePrice']]

train_data2_y.head()
train_data2_x = train_data2[['MSSubClass','YearBuilt','YearRemodAdd','BsmtUnfSF','1stFlrSF','2ndFlrSF','TotRmsAbvGrd','CentralAir']] #全テストデータの中から対象のカラムのみ抜き出す

train_data2_x.head()
lr = LinearRegression()

lr.fit(train_data2_x,train_data2_y)
print(lr.coef_[0])
print(lr.intercept_[0])
predict_train = pd.DataFrame(lr.predict(train_data2_x))
predict_train.head()
pd.concat([predict_train,train_data2_y],axis=1)
sub_train = train_data2_y.sub(predict_train)
sub_train
predict_test = pd.DataFrame(lr.predict(test_data2))
#提出用データの作成

submission = pd.DataFrame({

        "Id": test_data["Id"],

        "SalePrice": predict_test[0]})



#表示

print(submission)



#CSVファイルに出力

submission.to_csv("titanic_submission.csv", index=False)