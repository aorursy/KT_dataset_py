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
df = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
df.head()

df.fillna(0,inplace = True)
df.isnull().sum()
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

reg = RandomForestRegressor()
x = df[df.columns[:-1]]
y = df.SalePrice
x = x.select_dtypes(exclude=['object'])#Remove categorial data
reg.fit(x,y)
fea_imp = pd.DataFrame(reg.feature_importances_,index=x.columns)
fea_imp.sort_values(by=0,ascending=False)
features = x[['OverallQual','GrLivArea','TotalBsmtSF','2ndFlrSF','LotArea','1stFlrSF','YearBuilt','OverallCond']]
labels = y

features_train ,features_test, labels_train,labels_test = train_test_split(features,labels,test_size=0.3)
len(features_test)
reg.fit(features_train,labels_train)
reg.score(features_test,labels_test)#Random Forest accuracy
from sklearn.ensemble import GradientBoostingRegressor
gd = GradientBoostingRegressor()
gd.fit(features_train,labels_train)
gd.score(features_test,labels_test)
final_model = GradientBoostingRegressor()
final_model.fit(features_train,labels_train)
test_features = test_data[['OverallQual','GrLivArea','TotalBsmtSF','2ndFlrSF','LotArea','1stFlrSF','YearBuilt','OverallCond']]
test_features
#predictions = final_model.predict(test_features)

test_features.fillna(test_features.mean(),inplace = True)
test_features.isnull().sum()
predictions = final_model.predict(test_features)


output = pd.DataFrame({'Id':test_data.Id,'SalePrice':predictions})
output
output.to_csv('submission.csv',index=False)