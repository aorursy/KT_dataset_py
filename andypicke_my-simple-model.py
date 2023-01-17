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
train = pd.read_csv('../input/train.csv')
train.head()
train.columns
my_features = ['LotArea','YearBuilt','1stFlrSF','BedroomAbvGr']
X = train[my_features]
y = train.SalePrice

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X, y, random_state=1)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(Xtrain,ytrain)


test = pd.read_csv('../input/test.csv')
test_features = test[my_features]
preds = rf.predict(test_features)
my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':preds})
sample = pd.read_csv('../input/sample_submission.csv')
sample.head()
my_submission.to_csv('submission.csv',index=False)
