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
import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR 

from sklearn.metrics import r2_score



file=pd.read_csv('CarPrice_Assignment.csv')

data = pd.DataFrame(file)



plt.scatter(data['carlength'],data['price'],s=40,marker='^')

plt.xlabel('Car Length')

plt.ylabel('Price')

plt.show()

plt.close()



plt.scatter(data['enginesize'],data['price'],s=40,marker='o')

plt.xlabel('Enginesize')

plt.ylabel('Price')

plt.show()

plt.close()



plt.scatter(data['peakrpm'],data['price'],s=40,marker='^')

plt.xlabel('Peak rpm')

plt.ylabel('Price')

plt.show()

plt.close()



le = LabelEncoder()

data['CarName'] = le.fit_transform(data['CarName'])

data['fueltype']= le.fit_transform(data['fueltype'])

data['aspiration'] = le.fit_transform(data['aspiration'])

data['doornumber'] = le.fit_transform(data['doornumber'])

data['carbody']    = le.fit_transform(data['carbody'])

data['drivewheel'] = le.fit_transform(data['drivewheel'])

data['enginelocation'] = le.fit_transform(data['enginelocation'])

data['enginetype'] = le.fit_transform(data['enginetype'])

data['cylindernumber'] = le.fit_transform(data['cylindernumber'])

data['fuelsystem'] = le.fit_transform(data['fuelsystem'])



X = data.iloc[:,2:25]

Y = data['price']



X_train, X_test, Y_train,Y_test = train_test_split(X,Y, random_state=42,test_size=0.25)



print('Prediction using Linear Regression ')

print('==================================')

lr= linear_model.LinearRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)

#print(Y_test, Y_pred)

print(('r2_score = ')  + str(r2_score(Y_test,Y_pred_lr)))



print('')

print('Prediction using Kernel Ridge')

print('=============================')

KR = KernelRidge(alpha=0.1)

KR.fit(X_train,Y_train)

Y_pred_KR = KR.predict(X_test)

print(('r2 score =') + str(r2_score(Y_test,Y_pred_KR)))



print('')

print('Prediction using Random Forest')

print('=============================')

RF = RandomForestRegressor(random_state=1,n_estimators=10)

RF.fit(X_train,Y_train)

Y_pred_RF = RF.predict(X_test)

print(('r2 score = ') + str(r2_score(Y_test,Y_pred_RF)))
