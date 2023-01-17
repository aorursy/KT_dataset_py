#Import libraries which is required for building model

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Get data

from sklearn.datasets import load_boston



boston = load_boston()
#convert sklearn dataset to dataframe

df = pd.DataFrame(boston.data)

df.columns = boston.feature_names
#Let's add our target column now

df['Target'] = boston.target
df.head(10)
#Let's check correlation between target variable i.e Target with other variables

plt.figure(figsize=(12,7))

plt.title('Correlation Matrix')

sns.heatmap(df.corr(),annot=True)
#Get X and Y for splitting

X = df.iloc[:,:-1]

y = df.iloc[:,-1]
#Let's split in 80-20 ratio

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)
# 1. Will begin our model exploration with Linear regression model
from sklearn.linear_model import LinearRegression 

lr = LinearRegression() 

lr.fit(X_train, y_train) 
# from equation y = mx +c, let's fetch m term for all attribute other than Target

lr.coef_
#c value from y = mx + c equation, it will vary slightly based on test_size and random_state selected while splitting

lr.intercept_
#Converting the coefficient values to a dataframe

coeff = pd.DataFrame([X_train.columns,lr.coef_]).T

coeff = coeff.rename(columns={0: 'Attribute', 1: 'Coefficients'})

coeff
# Let's check our model on training part which is 80% of whole data

y_pred_train = lr.predict(X_train)
#Evaluation of model on training data

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print('R2:', r2_score(y_train,y_pred_train))

print('MAE:', mean_absolute_error(y_train,y_pred_train))

print('MSE:', mean_squared_error(y_train,y_pred_train))

print('RMSE:', np.sqrt(mean_squared_error(y_train,y_pred_train)))
#Plot a graph for actual price vs predicted price for training data

plt.scatter(y_train, y_pred_train, c = 'blue') 

plt.xlabel("Price: in $1000's") 

plt.ylabel("Predicted value") 

plt.title("True value vs predicted value : Linear Regression") 

plt.show() 
#Residual Plot

plt.scatter(y_pred_train,y_train-y_pred_train, c = 'blue')

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
#From grapgh its been clearly seen that residuals are equally distibuted around zero so our choice for selecting regression for this model is good

y_pred_test = lr.predict(X_test)
#Evaluation of model on 20% test data

print('R2:', r2_score(y_test,y_pred_test))

print('MAE:', mean_absolute_error(y_test,y_pred_test))

print('MSE:', mean_squared_error(y_test,y_pred_test))

print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred_test)))

lr_r2_score = r2_score(y_test,y_pred_test)
#R2 of training data and R2 of test data are not varying that much so our model is not overfitting

#As linear regression is giving around 70% of accuracy, let's try with Random forest regressor
# 2. Import Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor



# Create a Random Forest Regressor

reg = RandomForestRegressor()



# Train the model using the training sets 

reg.fit(X_train, y_train)
#Predicting training data using RFR model

y_pred_train1 = reg.predict(X_train)
#Evaluation based on RFR Model for training data

print('R2:', r2_score(y_train,y_pred_train1))

print('MAE:', mean_absolute_error(y_train,y_pred_train1))

print('MSE:', mean_squared_error(y_train,y_pred_train1))

print('RMSE:', np.sqrt(mean_squared_error(y_train,y_pred_train1)))
#Plot a graph for actual price vs predicted price for training data

plt.scatter(y_train, y_pred_train1, c = 'red') 

plt.xlabel("Price: in $1000's") 

plt.ylabel("Predicted value") 

plt.title("True value vs predicted value : Random Forest Regression") 

plt.show()
#Residual Plot

plt.scatter(y_pred_train1,y_train-y_pred_train1, c = 'red')

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
#Prediction using RFR Model on test data

y_pred_test1 = reg.predict(X_test)
#Evaluation based on RFR Model for test data

print('R2:', r2_score(y_test,y_pred_test1))

print('MAE:', mean_absolute_error(y_test,y_pred_test1))

print('MSE:', mean_squared_error(y_test,y_pred_test1))

print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred_test1)))

rfr_r2_score = r2_score(y_test,y_pred_test1)
#Wow, we have beaten linear regression's accuracy of 70% and using RFR we got 85% accuracy, let's explore other models also, lets hope we can beat RFR now
#3. Import XGBoost Regressor

from xgboost import XGBRegressor

xgb = XGBRegressor()



# Train the model using the training sets 

xgb.fit(X_train, y_train)
#Prediction using XGBoost Model on train data

y_pred_train2 = xgb.predict(X_train)
#Evaluation based on XGBoost Model for train data

print('R2:', r2_score(y_train,y_pred_train2))

print('MAE:', mean_absolute_error(y_train,y_pred_train2))

print('MSE:', mean_squared_error(y_train,y_pred_train2))

print('RMSE:', np.sqrt(mean_squared_error(y_train,y_pred_train2)))
#Plot a graph for actual price vs predicted price for training data

plt.scatter(y_train, y_pred_train2, c = 'green') 

plt.xlabel("Price: in $1000's") 

plt.ylabel("Predicted value") 

plt.title("True value vs predicted value : XGBoost") 

plt.show()
#Residual Plot

plt.scatter(y_pred_train2,y_train-y_pred_train2, c = 'green')

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
#Prediction using XGBoost Model on test data

y_pred_test2 = xgb.predict(X_test)
#Evaluation based on XGBoost Model for test data

print('R2:', r2_score(y_test,y_pred_test2))

print('MAE:', mean_absolute_error(y_test,y_pred_test2))

print('MSE:', mean_squared_error(y_test,y_pred_test2))

print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred_test2)))

xgb_r2_score = r2_score(y_test,y_pred_test2)
#ohhhh, that was really very close by margin xgb failed to beat RFR, anyways now lets try with last model SVR, but this model will require feature scaling
#4. SVR model importing and calling scaler

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler



#Prior to fitting model scale it



sc = StandardScaler()



#Fit only trained data, if we fit test data it will cause data leakage, transform can be done for both

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Fit model using SVR

svr = SVR(kernel = 'rbf')

svr.fit(X_train, y_train)
#Prediction using SVR Model on train data

y_pred_train3 = svr.predict(X_train)
#Evaluation based on SVR Model for train data

print('R2:', r2_score(y_train,y_pred_train3))

print('MAE:', mean_absolute_error(y_train,y_pred_train3))

print('MSE:', mean_squared_error(y_train,y_pred_train3))

print('RMSE:', np.sqrt(mean_squared_error(y_train,y_pred_train3)))
#Plot a graph for actual price vs predicted price for training data

plt.scatter(y_train, y_pred_train3, c = 'purple') 

plt.xlabel("Price: in $1000's") 

plt.ylabel("Predicted value") 

plt.title("True value vs predicted value : SVR") 

plt.show()
#Residual Plot

plt.scatter(y_pred_train2,y_train-y_pred_train3, c = 'purple')

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
#Prediction using SVR Model on test data

y_pred_test3 = svr.predict(X_test)
#Evaluation based on SVR Model for test data

print('R2:', r2_score(y_test,y_pred_test3))

print('MAE:', mean_absolute_error(y_test,y_pred_test3))

print('MSE:', mean_squared_error(y_test,y_pred_test3))

print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred_test3)))

svr_r2_score = r2_score(y_test,y_pred_test3)
#As we can see SVR is performing poor as compared to other models with lowest accuracy of 63% amongst all other models
#Dataframe for models accuracy comparison



acc_comparison = pd.DataFrame({'Model': ['Linear Regression', 'Random Forest Regressor', 'XGBoost', 'SVR'], 'R2 Score': [lr_r2_score*100, rfr_r2_score*100, xgb_r2_score*100, svr_r2_score*100]})



acc_comparison
#Let's conclude by saying RFR fits best for this dataset, One can also opt for XGBoost model