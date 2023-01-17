# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from sklearn.cross_validation import ShuffleSplit

#%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

df=pd.read_csv('../input/housing.csv',sep=",")
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.pairplot(df,size=2)
df.corr()  #for finding best feature
plt.figure(figsize=(16,10))

sns.heatmap(df.corr(),annot=True);
X = df[['LSTAT','RM','PTRATIO']] #select feature

y = df[['MEDV']].values   #select target var

y = y.reshape(-1,1)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)
#create linear regression object

lm = LinearRegression()  
#train the model using training set

lm.fit(X_train,y_train)
#make prediction using the training set first

y_train_pred = lm.predict(X_train)

y_test_pred = lm.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error



#the mean squared error,lower the value better, if it is a .0 means perfect prediction

s = mean_squared_error(y_train,y_train_pred)

print("Mean Squared error of training set :%2f"%s)
#the mean squared error,lower the value better it is .0 means perfect prediction

s = mean_squared_error(y_test,y_test_pred)

print("Mean squared error of testing set: %.2f"%s)
from sklearn.metrics import r2_score



# Explained variance score: 1 is perfect prediction

s = r2_score(y_train, y_train_pred)

print('R2 variance score of training set: %.2f' %s )
#explained the variance score :1 is perfect prediction

s = r2_score(y_test,y_test_pred)

print("R2 variance score of testing set: %2f"%s)
#calculating adjusted r2

N = y_test.size

p = X_train.shape[1]

adjr2score = 1 - ((1-r2_score(y_test, y_test_pred))*(N - 1))/ (N - p - 1)

print("Adjusted R^2 Score %.2f" % adjr2score)
#import polynomial package

from sklearn.preprocessing import PolynomialFeatures
#creat a polynomial regression model for the given degree=2

poly_reg = PolynomialFeatures(degree = 2)
#transform the existing feature to high degree features.

X_train_poly = poly_reg.fit_transform(X_train)

X_test_poly = poly_reg.fit_transform(X_test)
#fit the transform features to linear regression

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_train_poly,y_train)
#predicting on training data set 

y_train_predict = lin_reg_2.predict(X_train_poly)

#predicting on testing data set

y_test_predict = lin_reg_2.predict(X_test_poly)
#evaluating the model on train dataset

rmse_train = np.sqrt(mean_squared_error(y_train,y_train_predict))

r2_train = r2_score(y_train,y_train_predict)

print("The model performance of training set")

print("---------------------------------------------")

print("RMSE of training set is{}".format(rmse_train))

print("R2 score of training set is{}".format(r2_train))
#evaluating model on test dataset

rmse_test = np.sqrt(mean_squared_error(y_test,y_test_predict))

r2_test = r2_score(y_test,y_test_predict)



print("The model performance of training set")

print("-----------------------------------------------")

print("RMSE of testing set is{}".format(rmse_test))

print("R2 score of testing set is{}".format(r2_test))
#import polynomial package

from sklearn.preprocessing import PolynomialFeatures
#creat a polynomial regression model for the given degree=3

poly_reg = PolynomialFeatures(degree = 3)
#transform the existing feature to high degree features.

X_train_poly = poly_reg.fit_transform(X_train)

X_test_poly = poly_reg.fit_transform(X_test)
#fit the transform features to linear regression

lin_reg_3 = LinearRegression()

lin_reg_3.fit(X_train_poly,y_train)
#predicting on training data set 

y_train_predict = lin_reg_3.predict(X_train_poly)

#predicting on testing data set

y_test_predict = lin_reg_3.predict(X_test_poly)
#evaluating the model on train dataset

rmse_train = np.sqrt(mean_squared_error(y_train,y_train_predict))

r2_train = r2_score(y_train,y_train_predict)

print("The model performance of training set")

print("----------------------------------------------")

print("RMSE of training set is{}".format(rmse_train))

print("R2 score of training set is{}".format(r2_train))
#evaluating model on test dataset

rmse_test = np.sqrt(mean_squared_error(y_test,y_test_predict))

r2_test = r2_score(y_test,y_test_predict)



print("The model performance of testing set")

print("--------------------------------------------")

print("RMSE of testing set is{}".format(rmse_test))

print("R2 score of testing set is{}".format(r2_test))
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)

y_std = sc_y.fit_transform(y.reshape(-1,1)).flatten()
X_std.shape
import numpy as np

alpha = 0.0001    #learning rate

w_ = np.zeros(1 + X_std.shape[1])    

cost_ = [] 

n_ = 100

 

for i in range(n_):

    y_pred = np.dot(X_std,w_[1:] + w_[0])

    errors  = (y_std - y_pred)

    

    w_[1:] +=alpha * X_std.T.dot(errors)   #theta1

    w_[0] +=alpha *errors.sum()        #theta0

    

    cost = (errors**2).sum() / 2.0

    cost_.append(cost)
plt.figure(figsize=(10,8))  #plot the figure

plt.plot(range(1,n_ + 1),cost_);

plt.ylabel('SSE');

plt.xlabel('Epoch');
w_   #gradient function (intercept and coeficient) 
#accuracy of gradient function

print("Accuracy: %0.2f (+/- %0.2f)" % (w_.mean(), w_.std() * 2))