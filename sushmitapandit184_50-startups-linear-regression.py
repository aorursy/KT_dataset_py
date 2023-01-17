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
df=pd.read_csv('../input/50_Startups.csv')
df.head()
#df.info()
X=df.iloc[:,:-1].values #predictor varables
y=df.iloc[:,4].values #outcome variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder()
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X=X[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

#extracting data from model
print('y intercepts:',model.intercept_) #prints y intercept
print('model coeff:',model.coef_) #prints the coefficients in the same order as they are passed

#predicting the test set results
y_pred=model.predict(X_test)
print('predictions:',y_pred)
#import metrics library
from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y_test,y_pred)) #calculating mean absolute error MAE

print('MSE',metrics.mean_squared_error(y_test,y_pred)) #calculating mean squared error MSE

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,y_pred))) #calulating root mean squared error RMSE
