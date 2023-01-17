# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

train_df = pd.read_csv('../input/housingdata/data_test.csv')
print(train_df.columns.values)

print(train_df.head())

reg = LinearRegression()

labels = train_df['SalePrice'];
conv_dates = [1 if values == 2010 else 0 for values in train_df.YrSold]
train_df['YrSold'] = conv_dates
train1 = train_df.drop(['Id','SalePrice'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(train1,labels,test_size=0.20, random_state=2)
reg.fit(x_train,y_train)
print("")
print(reg.score(x_test,y_test))
print("")

