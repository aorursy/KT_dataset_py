# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
home=pd.read_csv('../input/train.csv')
home=home[(home.Id != 1183) & (home.Id !=692)]
#print(home.describe)
#home.describe
y=home.SalePrice
#home[Features].isnull().any()
home['Age']=home['YrSold']-home['YearBuilt']
Features=['LotArea','Age','OverallQual', 'LotShape','LandContour','LandSlope',
          'Neighborhood','Condition1','BldgType','HouseStyle']
X=pd.get_dummies(home[Features],columns=['LotShape','LandContour','LandSlope',
          'Neighborhood','Condition1','BldgType','HouseStyle'], prefix='col')
#X.columns
X=X.drop(['col_2.5Fin'],axis=1)
#print(X.columns)
model=LinearRegression()
model.fit(X,y)
#scores = cross_val_score(model,X,y,cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print(scores)
#plt.scatter(home.Age,y)
#plt.bar(home.Street,y)  only 6 values of grvl
#dummies : LotShape,LandContour,LandSlope,Neighborhood,Condition1,BldgType
#lable encoding : HouseStyle,
test = pd.read_csv('../input/test.csv')
test['Age']=test['YrSold']-test['YearBuilt']
test_X=pd.get_dummies(test[Features],columns=['LotShape','LandContour','LandSlope',
          'Neighborhood','Condition1','BldgType','HouseStyle'], prefix='col')
#print(test_X.columns)
model.predict(test_X)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': model.predict(test_X)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)