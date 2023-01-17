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
#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
cars = pd.read_csv("../input/autos.csv",encoding='latin1')
cars.head()
cars.isnull().sum()
cars_updated = cars.dropna()
cars_updated.isnull().sum()
cars_updated = cars_updated.iloc[:,[6,7,8,9,10,11,12,13,14,15,4]]
cars_updated.columns
print('Vehicle Type: ',cars_updated.vehicleType.unique())
print('Gearbox: ',cars_updated.gearbox.unique())
print('Fuel Type: ',cars_updated.fuelType.unique())
print('Repaired Damage: ',cars_updated.notRepairedDamage.unique())
cars_updated.replace({'gearbox':{'manuell':'manual','automatik':'automatic'}},inplace=True)
cars_updated.replace({'vehicleType':{'kleinwagen':'small_car','kombi':'combi','andere':'Others'}},inplace=True)
cars_updated.replace({'fuelType':{'benzin':'petrol','andere':'others','elektro':'electro'}},inplace=True)
cars_updated.replace({'notRepairedDamage':{'nein':'no','ja':'yes'}},inplace=True)
cars_updated.head(10)
cars_updated = cars_updated.loc[(cars_updated.price>400)&(cars_updated.price<=40000)]
cars_updated = cars_updated.loc[(cars_updated.yearOfRegistration>1990)&(cars_updated.yearOfRegistration<=2016)]
cars_updated = cars_updated.loc[(cars_updated.powerPS>10)]
cars_updated = cars_updated.loc[(cars_updated.kilometer>1000)&(cars_updated.kilometer<=150000)]
#Replacing all the 0 month values to 1
cars_updated.monthOfRegistration.replace(0,1,inplace=True)
# Making the year and month column to get a single date
Purchase_Datetime=pd.to_datetime(cars_updated.yearOfRegistration*10000+cars_updated.monthOfRegistration*100+1,format='%Y%m%d')
import time
from datetime import date
y=date(2018, 5,1)
# Calculating days old by subracting both date fields and converting them into integer
Days_old=(y-Purchase_Datetime)
Days_old=(Days_old / np.timedelta64(1, 'D')).astype(int)
#type(Days_old[1])
cars_updated['Days_old']=Days_old
cars_updated.drop(columns=['yearOfRegistration','monthOfRegistration','powerPS'],inplace=True)
cars_dummies=pd.get_dummies(data=cars_updated,columns=['notRepairedDamage','vehicleType','model','brand','gearbox','fuelType'])
cars_dummies.head(10)
X = cars_dummies.drop('price',axis=1)

y = cars_dummies.price
X.head(5)
y.head(5)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)
print (linreg.intercept_)
# pair the feature names with the coefficients
list(zip(X.columns.get_values(), linreg.coef_))
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=123)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#Predicting the test set results
y_pred = linreg.predict(X_test)
print(linreg.score(X_test, y_test)*100,'% Prediction Accuracy')