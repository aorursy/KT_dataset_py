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
train_file = pd.read_csv("../input/train.csv")
test_train_file = pd.read_csv("../input/test.csv")
train_file.head()
train_file.fillna(0,inplace = True)
train_file.isnull().sum()

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

mo = RandomForestRegressor()
x = train_file[train_file.columns[:-1]]
y = train_file.SalePrice
x = x.select_dtypes(exclude=['object'])
mo.fit(x,y)
fea_imp = pd.DataFrame(mo.feature_importances_,index=x.columns)
fea_imp.sort_values(by=0,ascending=False)


fea_imp = pd.DataFrame(mo.feature_importances_,index=x.columns)
fea_imp.sort_values(by=0,ascending=False)

features = x[['OverallQual','GrLivArea','TotalBsmtSF','2ndFlrSF','LotArea','1stFlrSF','YearBuilt','OverallCond']]
labels = y

features_train ,features_test, labels_train,labels_test = train_test_split(features,labels,test_size=0.4)
len(features_test)
mo.fit(features_train,labels_train)
mo.score(features_test,labels_test)#Random Forest accuracy

from sklearn.ensemble import GradientBoostingRegressor
moh = GradientBoostingRegressor()
moh.fit(features_train,labels_train)
moh.score(features_test,labels_test)

final_model = GradientBoostingRegressor()
final_model.fit(features_train,labels_train)
test_features = test_train_file[['OverallQual','GrLivArea','TotalBsmtSF','2ndFlrSF','LotArea','1stFlrSF','YearBuilt','OverallCond']]
test_features
test_features.fillna(test_features.mean(),inplace = True)
test_features.isnull().sum()

predictions = final_model.predict(test_features)

output = pd.DataFrame({'Id':test_train_file.Id,'SalePrice':predictions})
output
output.to_csv('../input/submission.csv',index=False)

