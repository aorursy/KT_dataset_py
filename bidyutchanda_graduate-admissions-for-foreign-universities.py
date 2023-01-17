import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dataset = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
dataset.info()
dataset.head()
X = dataset.iloc[:, 1:8] #All rows and all columns except first and last ones are independent variables. 

X.head()
y = dataset.iloc[:, 8] #Only the last column in our data is the dependent variable column. 

y.head()
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 1/5, random_state=0)
#Fitting SVR for the training sets

from sklearn.svm import SVR

regressor_SVR = SVR (kernel ='linear') #default kernel is 'rbf'.

regressor_SVR.fit(train_X, train_y)



#Making the necessary predictions

preds = regressor_SVR.predict(test_X)



#Measuring the accuracy

from sklearn.metrics import r2_score,mean_absolute_error

print(f'R^2 score: {r2_score(test_y, preds):.2f}')

print(f'MAE: {mean_absolute_error(test_y, preds):.2f}')

#Fitting Decision Tree Regressor to training sets

from sklearn.tree import DecisionTreeRegressor

regressor_Tree = DecisionTreeRegressor (random_state = 0)

regressor_Tree.fit (train_X, train_y)



#Making the necessary predictions

preds_Tree = regressor_Tree.predict(test_X)



#Measuring the accuracy

from sklearn.metrics import r2_score,mean_absolute_error

print(f'R^2 score: {r2_score(test_y, preds_Tree):.2f}')

print(f'MAE: {mean_absolute_error(test_y, preds_Tree):.2f}')

#Fitting Random Forest Regressor to training sets

from sklearn.ensemble import RandomForestRegressor

regressor_Forest = RandomForestRegressor (random_state = 0, n_estimators = 50)

#n_estimators is the number of trees we want to generate. 

#For n_estimators=10, R^2 score: 0.73; MAE:0.05

#For n_estimators=20, R^2 score: 0.75; MAE:0.05

#For n_estimators=30, R^2 score: 0.76; MAE:0.04

#For n_estimators=40, R^2 score: 0.77; MAE:0.04

#For n_estimators=50, R^2 score: 0.76; MAE:0.04

#So we see that after 40 trees, the R^2 score decreases and hence the most optimum no. of trees is 40, which gives the best predictions. 

regressor_Forest.fit (train_X, train_y)



#Making the necessary predictions

preds_Forest = regressor_Forest.predict(test_X)



#Measuring the accuracy

from sklearn.metrics import r2_score,mean_absolute_error

print(f'R^2 score: {r2_score(test_y, preds_Forest):.2f}')

print(f'MAE: {mean_absolute_error(test_y, preds_Forest):.2f}')
from xgboost.sklearn import XGBClassifier

classifier_XGB = XGBClassifier()

classifier_XGB.fit(train_X, train_y)



preds_XGB = classifier_XGB.predict(test_X)



from sklearn.metrics import r2_score,mean_absolute_error

print(f'R^2 score: {r2_score(test_y, preds_Forest):.2f}')

print(f'MAE: {mean_absolute_error(test_y, preds_Forest):.2f}')