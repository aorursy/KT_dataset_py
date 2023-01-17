

# Importing Libraries 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 



# Importing Data 

from sklearn.datasets import load_boston 

boston = load_boston() 

#print(boston.DESCR)

boston.feature_names





boston.data.shape 

data = pd.DataFrame(boston.data) 

data.columns = boston.feature_names 



data.head(10) 

#Adding price column

boston.target.shape

data['PRICE']= boston.target

data.head()

# Input Data 

x = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',

       'PTRATIO', 'B', 'LSTAT']]



# Output Data 

y = boston.target 





# splitting data to training and testing dataset. 

from sklearn.model_selection import train_test_split 

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2, 

													random_state = 0) 



print("xtrain shape : ", xtrain.shape) 

print("xtest shape : ", xtest.shape) 

print("ytrain shape : ", ytrain.shape) 

print("ytest shape : ", ytest.shape) 

# Fitting Multi Linear regression model to training model 

from sklearn.linear_model import LinearRegression 

regressor = LinearRegression() 

regressor.fit(xtrain, ytrain) 



# predicting the test set results 

y_pred = regressor.predict(xtest) 

regressor.coef_
import seaborn as sns

sns.distplot(ytest-y_pred, bins=30)
coeff_df = pd.DataFrame(regressor.coef_,x.columns,columns=['Coefficient'])

coeff_df

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(ytest, y_pred))

print('MSE:', metrics.mean_squared_error(ytest, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, y_pred)))