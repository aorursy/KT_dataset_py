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
train = pd.read_csv('../input/train_V2.csv')
train = train[train['winPlacePerc'].isna() == False]
features = train.drop(['winPlacePerc','killPoints','killPlace','DBNOs','headshotKills','rideDistance'] , axis = 1)._get_numeric_data()  
#train[['maxPlace','winPoints','rankPoints','killPoints','killPlace','DBNOs','headshotKills','rideDistance']]
target   = train[['winPlacePerc']]

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size = 0.3)
#import linear regression classifier, initialize and fit the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

y_pred = regressor.predict(X_test)
# The next is to evaluate the classifier using metrics such as the mean square error 
# and the coefficient of determination R square

from sklearn.metrics import mean_squared_error,r2_score

#The coefficients 
print(features.columns.values)
print('Coefficients: \n', regressor.coef_)

#The mean squared error
print('The mean squared error: {:2f}'.format(mean_squared_error(y_test , y_pred)))

#Explained variance score: 1 is perfect prediction
print('Variance score: {:.2f}'.format(r2_score(y_test ,y_pred)))


test = pd.read_csv('../input/test_V2.csv')
test_pred = regressor.predict(test.drop(['killPoints','killPlace','DBNOs','headshotKills','rideDistance'], axis = 1)._get_numeric_data())

submission_df = pd.concat([test['Id'], pd.DataFrame(test_pred)], axis=1, sort=False )
submission_df.columns = ['Id','winPlacePerc']

submission_df.columns
submission_df.to_csv('submission.csv',index =False)
