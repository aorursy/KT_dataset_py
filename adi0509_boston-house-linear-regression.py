import numpy as np

import seaborn as sns

import pandas as pd



from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import train_test_split
# Importing Data 

from sklearn.datasets import load_boston 

boston_data = load_boston() 
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

boston.head()
#Now we'll add price column

boston['Price'] = boston_data.target

boston.head() 
boston.describe()
boston.info()
# splitting data to training and testing dataset



# Input Data 

x = boston_data.data 

   

# Output Data 

y = boston_data.target 



xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size =0.2, 

                                                    random_state = 0) 

   

print("xtrain shape : ", xtrain.shape) 

print("xtest shape  : ", xtest.shape) 

print("ytrain shape : ", ytrain.shape) 

print("ytest shape  : ", ytest.shape) 
# Fitting Multi Linear regression model to training model 

lr = LinearRegression() 

lr.fit(xtrain, ytrain)



# predicting the test set results 

y_pred = lr.predict(xtest) 
# Results of Linear Regression. 



from sklearn import metrics

from sklearn.metrics import r2_score

print('Mean Absolute Error : ', metrics.mean_absolute_error(ytest, y_pred))

print('Mean Square Error : ', metrics.mean_squared_error(ytest, y_pred))

print('RMSE', np.sqrt(metrics.mean_squared_error(ytest, y_pred)))

print('R squared error', r2_score(ytest, y_pred))
#Actual Value Vs Predicted Value

df1 = pd.DataFrame({'Actual': ytest, 'Predicted':y_pred})

df2 = df1.head(10)

df2.plot(kind = 'bar')
sns.pairplot(boston)
#np.c_: it is used to concatenate columns

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])

Y = boston['Price']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size =0.2, 

                                                    random_state = 0) 

   

print("xtrain shape : ", Xtrain.shape) 

print("xtest shape  : ", Xtest.shape) 

print("ytrain shape : ", Ytrain.shape) 

print("ytest shape  : ", Ytest.shape) 
# Fitting Multi Linear regression model to training model 

lr = LinearRegression() 

lr.fit(Xtrain, Ytrain)



# predicting the test set results 

Y_pred = lr.predict(Xtest)
# Results of Linear Regression.  



from sklearn import metrics

from sklearn.metrics import r2_score

print('Mean Absolute Error : ', metrics.mean_absolute_error(Ytest, Y_pred))

print('Mean Square Error : ', metrics.mean_squared_error(Ytest, Y_pred))

print('RMSE', np.sqrt(metrics.mean_squared_error(Ytest, Y_pred)))

print('R squared error', r2_score(Ytest, Y_pred))