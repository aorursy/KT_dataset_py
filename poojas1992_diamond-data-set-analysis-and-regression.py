import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
#import dataset using pandas

data_diamond = pd.read_csv("../input/diamonds.csv")

data_diamond.head()
#Dropping the id feature

data_diamond.drop(columns = ['Unnamed: 0'], inplace = True)

data_diamond.head()
#converting non-numeric values to numeric

data_diamond['cut'],_ = pd.factorize(data_diamond['cut'])  

data_diamond['color'],_ = pd.factorize(data_diamond['color'])  

data_diamond['clarity'],_ = pd.factorize(data_diamond['clarity'])  

data_diamond.head()
mask = np.zeros_like(data_diamond.corr())

mask[np.triu_indices_from(mask)] = True



fig, ax = plt.subplots()

fig.set_size_inches(14, 10)



ax = sns.heatmap(data_diamond.corr(), annot = True, mask = mask)
#Spliting data into test and training set 

Training_Data ,Test_Data = train_test_split(data_diamond,test_size=0.1)
#Creating training and test datasets for Regression

X_train = Training_Data.drop(["price"], axis = 1)

y_train = Training_Data.price

X_test = Test_Data.drop(["price"], axis = 1)

y_test = Test_Data.price
#Linear Regression

Regression_Linear = LinearRegression()

Regression_Linear.fit(X_train , y_train)

y_pred_LiR = Regression_Linear.predict(X_test)



print('####### Linear Regression #######')

print('Accuracy : %.4f' % Regression_Linear.score(X_test, y_test))



print('Coefficients: \n', Regression_Linear.coef_)

print('')

print('MSE    : %0.2f ' % mean_squared_error(y_test, y_pred_LiR))

print('R2     : %0.2f ' % r2_score(y_test, y_pred_LiR))
#Random Forest Regression

Regression_RF = RandomForestRegressor()

Regression_RF.fit(X_train , y_train)



y_pred_RF = Regression_RF.predict(X_test)



print('###### Random Forest ######')

print('Accuracy : %.4f' % Regression_RF.score(X_test, y_test))



print('Coefficients: \n', Regression_Linear.coef_)

print('')

print('MSE    : %0.2f ' % mean_squared_error(y_test, y_pred_RF))

print('R2     : %0.2f ' % r2_score(y_test, y_pred_RF))