#The following codes trains a regression model to predict the profits of a set of startup companies in the US.

#The models tested are Linear Regression and Decision Tree Regressor.

#An analysis is done to determine which model is a more accurate model for this application.



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#this line is to enable auto-prediction of commands

%config IPCompleter.greedy=True
DT_startups_df=pd.read_csv("../input/50_Startups.csv")
DT_startups_df.head()
iv=DT_startups_df[['R&D Spend','Administration','Marketing Spend','State']]
dv=DT_startups_df[['Profit']]
#use OneHotEncoding to deal with the State string variable

iv=pd.get_dummies(iv,drop_first=True)
#split data into training and test sets

from sklearn.model_selection import train_test_split

iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)
iv_test
#Create a Linear regression model and fit it to the training set (iv_train and dv_train)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(iv_train,dv_train)
#predict the Profit values using this trained Linear Regression model, using the test set of independent variables iv_test

y_pred_lr=lr.predict(iv_test)
#create a table to compare the actual test set Profit values to the predicted Profit values using linear regression

CompareTable=pd.DataFrame()

CompareTable['Actual Profit']=pd.Series(range(len(dv_test)))

CompareTable['Actual Profit']=pd.DataFrame(dv_test).reset_index(drop=True) #if you do not put drop=True, then the actual index values will show

CompareTable['LinReg Predicted Profit']=pd.DataFrame(y_pred_lr).reset_index(drop=True)

CompareTable.head()
#Create a Decision Tree Regression Model and fit it to the training set (iv_train and dv_train)

from sklearn.tree import DecisionTreeRegressor

dcr=DecisionTreeRegressor(max_depth=3)

dcr.fit(iv_train,dv_train)
#predict the Profit values using this trained Decision Tree Regression model, using the test set of independent variables iv_test

profit_pred_dcr=dcr.predict(iv_test)
#add a new column to the Comparison table, showing the predicted Profit values using Decision Tree Regressor

CompareTable['DT Predicted Profit']=pd.DataFrame(profit_pred_dcr).reset_index(drop=True)

CompareTable.head(10)
#check for the goodness of fit for both Linear Regression and Decision Tree Regression models

from sklearn.metrics import mean_squared_error



rmse_lr = np.sqrt(mean_squared_error(dv_test,y_pred_lr))

rmse_dcr = np.sqrt(mean_squared_error(dv_test,profit_pred_dcr))

print("root mean squared error using Linear Regression:",rmse_lr)

print("root mean squared error using DecisionTree Regression:",rmse_dcr)



if (rmse_lr<rmse_dcr):

   print("Linear regression is more accurate prediction model than decision tree regression for this dataset.")

else:

   print("Decision Tree regression is more accurate prediction model than Linear regression for this dataset.")