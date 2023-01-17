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
data1 = pd.read_csv('../input/train.csv')

data2 = pd.read_csv('../input/test.csv')

data = pd.concat([data1, data2])
#missing values treatment

data.isnull().sum()



#make a dummy variable for NA in : alley, FireplaceQu GarageType GarageYrBlt GarageFinish GarageCars GarageArea GarageQual GarageCond 

# BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinSF1 BsmtFinType2 

#since the documentation tells us that NA is supposed to happen and has an actual meaning



#lotfrontage : replace by average

data.LotFrontage.fillna(data.LotFrontage.mean(), inplace=True)



#Electrical : only one case, just take standard type for this one

data.Electrical.fillna('SBrkr', inplace = True)



#consider no garage equal to very old one

data.GarageYrBlt.fillna(data.GarageYrBlt.min(), inplace = True)



#masvnrarea has many 0 values, lets put NA to 0

data.MasVnrArea.fillna(0, inplace = True)
#convert the data to dummies (avoid categorical data)

data_dummy = pd.get_dummies(data, dummy_na=True)
#drop id column,  dont want it in the modeling data

data_dummy.drop('Id', axis = 1, inplace = True)
x_length = data1.shape[0]

training = data_dummy.iloc[0:x_length, :]

preding = data_dummy.iloc[x_length:, :]
X = training.drop('SalePrice', axis = 1).values

y = training.SalePrice.values
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
parameters = {'n_estimators': [10, 50, 100], 'max_features': [0.8, 0.9, 0.95]}

RF = GridSearchCV(RandomForestRegressor(random_state=0), parameters, cv=5)

RF.fit(X_train, y_train)
#mean average percent error is 0.9 -> quite good result

1 - (abs(y_test - RF.predict(X_test)) / y_test).mean()
predos = RF.predict(preding.drop('SalePrice', axis = 1).fillna(0).values)

result = pd.concat([data2.Id, pd.Series(predos, name = 'SalePrice')], axis = 1)

result.to_csv('house_predos_joos.csv', index = False)