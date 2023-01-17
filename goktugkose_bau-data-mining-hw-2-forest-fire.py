import os

import pandas as pd

import numpy as np

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

print(os.listdir("../input"))
df_results= pd.DataFrame(index=['Lasso Regression','Ridge Regression','Linear Regression']

                         ,columns=['MAE','RMSE','R^2'])

df_scores = pd.DataFrame(index=['Lasso Regression', 'Ridge Regression','Linear Regression'],

                         columns=['Training Score', 'Testing Score','# of Coefficients Used'])
fire = pd.read_csv('../input/forestfires.csv')

print('There are', fire.shape[0], 'rows and', fire.shape[1], 'columns in the dataset.')
# X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain --> Predictors (x)

# area --> Target (y)

fire.columns
fire.head()
# Encoding the string values with LabelEncoder()

le = LabelEncoder()

fire['month'] = le.fit_transform(fire['month'])

fire['day'] = le.fit_transform(fire['day'])
fire.head()
# Splitting the data and target columns of the data.

X = fire.iloc[:,:-1]

y = fire.iloc[:,-1]
X.head()
#Splitting dataset into train and test datasets.

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
#Applying Lasso Regression to the dataset.

lasso = Lasso()

lasso.fit(X_train,y_train)



train_score_lasso = lasso.score(X_train,y_train)

test_score_lasso = lasso.score(X_test,y_test)

coeff_used_lasso = np.sum(lasso.coef_!=0)



df_scores.iloc[0][0] = train_score_lasso

df_scores.iloc[0][1] = test_score_lasso 

df_scores.iloc[0][2] = coeff_used_lasso 



pred_lasso = lasso.predict(X_test)
#Measuring the performance of the Lasso Regression

mae_lasso = mean_absolute_error(y_test,pred_lasso)

mse_lasso = mean_squared_error(y_test,pred_lasso)

r2_lasso = r2_score(y_test,pred_lasso)



df_results.iloc[0][0] = mae_lasso

df_results.iloc[0][1] = mse_lasso

df_results.iloc[0][2] = r2_lasso
#Applying Ridge Regression to the dataset.

ridge = Ridge()

ridge.fit(X_train,y_train)



train_score_ridge = ridge.score(X_train,y_train)

test_score_ridge = ridge.score(X_test,y_test)

coeff_used_ridge = np.sum(ridge.coef_!=0)



df_scores.iloc[1][0] = train_score_ridge

df_scores.iloc[1][1] = test_score_ridge 

df_scores.iloc[1][2] = coeff_used_ridge 



pred_ridge = ridge.predict(X_test)
#Measuring the performance of the Ridge Regression

mae_ridge = mean_absolute_error(y_test,pred_ridge)

mse_ridge = mean_squared_error(y_test,pred_ridge)

r2_ridge = r2_score(y_test,pred_ridge)



df_results.iloc[1][0] = mae_ridge

df_results.iloc[1][1] = mse_ridge

df_results.iloc[1][2] = r2_ridge
linear = LinearRegression()

linear.fit(X_train,y_train)



train_score_linear = linear.score(X_train,y_train)

test_score_linear = linear.score(X_test,y_test)

coeff_used_linear = np.sum(linear.coef_!=0)



df_scores.iloc[2][0] = train_score_linear

df_scores.iloc[2][1] = test_score_linear 

df_scores.iloc[2][2] = coeff_used_linear 



pred_linear = linear.predict(X_test)
#Measuring the performance of the Linear Regression

mae_linear = mean_absolute_error(y_test,pred_linear)

mse_linear = mean_squared_error(y_test,pred_linear)

r2_linear = r2_score(y_test,pred_linear)



df_results.iloc[2][0] = mae_linear

df_results.iloc[2][1] = mse_linear

df_results.iloc[2][2] = r2_linear
df_results
df_scores