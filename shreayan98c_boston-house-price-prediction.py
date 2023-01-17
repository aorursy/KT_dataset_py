# Importing the libraries 

import pandas as pd

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing the Boston Housing dataset

from sklearn.datasets import load_boston

boston = load_boston()
# Initializing the dataframe

data = pd.DataFrame(boston.data)
# See head of the dataset

data.head()
#Adding the feature names to the dataframe

data.columns = boston.feature_names

data.head()
#Adding target variable to dataframe

data['PRICE'] = boston.target 

# Median value of owner-occupied homes in $1000s
#Check the shape of dataframe

data.shape
data.columns
data.dtypes
# Identifying the unique number of values in the dataset

data.nunique()
# Check for missing values

data.isnull().sum()
# See rows with missing values

data[data.isnull().any(axis=1)]
# Viewing the data statistics

data.describe()
# Finding out the correlation between the features

corr = data.corr()

corr.shape
# Plotting the heatmap of correlation between features

plt.figure(figsize=(20,20))

sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
# Spliting target variable and independent variables

X = data.drop(['PRICE'], axis = 1)

y = data['PRICE']
# Splitting to training and testing data



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)
# Import library for Linear Regression

from sklearn.linear_model import LinearRegression



# Create a Linear regressor

lm = LinearRegression()



# Train the model using the training sets 

lm.fit(X_train, y_train)
# Value of y intercept

lm.intercept_
#Converting the coefficient values to a dataframe

coeffcients = pd.DataFrame([X_train.columns,lm.coef_]).T

coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})

coeffcients
# Model prediction on train data

y_pred = lm.predict(X_train)
# Model Evaluation

print('R^2:',metrics.r2_score(y_train, y_pred))

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_train, y_pred))

print('MSE:',metrics.mean_squared_error(y_train, y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
# Visualizing the differences between actual prices and predicted values

plt.scatter(y_train, y_pred)

plt.xlabel("Prices")

plt.ylabel("Predicted prices")

plt.title("Prices vs Predicted prices")

plt.show()
# Checking residuals

plt.scatter(y_pred,y_train-y_pred)

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
# Checking Normality of errors

sns.distplot(y_train-y_pred)

plt.title("Histogram of Residuals")

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.show()
# Predicting Test data with the model

y_test_pred = lm.predict(X_test)
# Model Evaluation

acc_linreg = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_linreg)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
# Import Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor



# Create a Random Forest Regressor

reg = RandomForestRegressor()



# Train the model using the training sets 

reg.fit(X_train, y_train)
# Model prediction on train data

y_pred = reg.predict(X_train)
# Model Evaluation

print('R^2:',metrics.r2_score(y_train, y_pred))

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_train, y_pred))

print('MSE:',metrics.mean_squared_error(y_train, y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
# Visualizing the differences between actual prices and predicted values

plt.scatter(y_train, y_pred)

plt.xlabel("Prices")

plt.ylabel("Predicted prices")

plt.title("Prices vs Predicted prices")

plt.show()
# Checking residuals

plt.scatter(y_pred,y_train-y_pred)

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
# Predicting Test data with the model

y_test_pred = reg.predict(X_test)
# Model Evaluation

acc_rf = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_rf)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
# Import XGBoost Regressor

from xgboost import XGBRegressor



#Create a XGBoost Regressor

reg = XGBRegressor()



# Train the model using the training sets 

reg.fit(X_train, y_train)
# Model prediction on train data

y_pred = reg.predict(X_train)
# Model Evaluation

print('R^2:',metrics.r2_score(y_train, y_pred))

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_train, y_pred))

print('MSE:',metrics.mean_squared_error(y_train, y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
# Visualizing the differences between actual prices and predicted values

plt.scatter(y_train, y_pred)

plt.xlabel("Prices")

plt.ylabel("Predicted prices")

plt.title("Prices vs Predicted prices")

plt.show()
# Checking residuals

plt.scatter(y_pred,y_train-y_pred)

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
#Predicting Test data with the model

y_test_pred = reg.predict(X_test)
# Model Evaluation

acc_xgb = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_xgb)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
# Creating scaled set to be used in model to improve our results

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Import SVM Regressor

from sklearn import svm



# Create a SVM Regressor

reg = svm.SVR()
# Train the model using the training sets 

reg.fit(X_train, y_train)
# Model prediction on train data

y_pred = reg.predict(X_train)
# Model Evaluation

print('R^2:',metrics.r2_score(y_train, y_pred))

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_train, y_pred))

print('MSE:',metrics.mean_squared_error(y_train, y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
# Visualizing the differences between actual prices and predicted values

plt.scatter(y_train, y_pred)

plt.xlabel("Prices")

plt.ylabel("Predicted prices")

plt.title("Prices vs Predicted prices")

plt.show()
# Checking residuals

plt.scatter(y_pred,y_train-y_pred)

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
# Predicting Test data with the model

y_test_pred = reg.predict(X_test)
# Model Evaluation

acc_svm = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_svm)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
models = pd.DataFrame({

    'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines'],

    'R-squared Score': [acc_linreg*100, acc_rf*100, acc_xgb*100, acc_svm*100]})

models.sort_values(by='R-squared Score', ascending=False)