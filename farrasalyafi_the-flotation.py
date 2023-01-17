# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Importing all libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data, so we have to convert the date column into date 
# and we have to drop the duplicates entries/row using code below

data =  pd.read_csv('../input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv',
                   decimal=",",
                    parse_dates=["date"],
                    infer_datetime_format=True).drop_duplicates()
data.info()
#Check the shape of data (row and column)
data.shape
#check the data if there any missing value or not
data.isnull().sum()
#Display the data, and observe what kind of the data is this
data.head()
#We use heatmap to visualize the corealtion between each features

plt.figure(figsize=(30, 30))
cor= data.corr()
corelation = sns.heatmap(cor, annot=True, cmap="RdYlGn")
#Drop data that there are no significant corelation on dependent feature
#Make Correlation with output variable
cor_target = abs(cor["% Silica Concentrate"])

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.15]
relevant_features
#We pick the 2 biggest corelation exclude target
relevant_features = relevant_features.nlargest(n=3)
#Make a data from the relevant features
data = pd.DataFrame(data, columns=relevant_features.index)
data.head()
#Checking The Outlier in our data
sns.boxplot(data['Flotation Column 01 Air Flow'])
#Checking The Outlier in our data
sns.boxplot(data['% Iron Concentrate'])
#Checking The Outlier in our data
sns.boxplot(data['% Silica Concentrate'])
#Dropping the outlier with Percentiles
for i in data:
    upper_lim = data[i].quantile(.95)
    lower_lim = data[i].quantile(.05)

    data = data[(data[i] < upper_lim) & (data[i] > lower_lim)]
# Before we split into train and test data, as we can see, the data have differents in units and magnitude
# So to make it at the same magnitude we can scaling the data

y = data['% Silica Concentrate']
X = data.drop(['% Silica Concentrate'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# After we scaled the data, and the data have the same magnitude
# we can split the data into Train & Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    y,
                                                    test_size=0.3,
                                                   random_state=30)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_linreg = lin_reg.predict(X_test)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)
y_pred_ridge = ridge_regressor.predict(X_test)
from sklearn.linear_model import Lasso

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
y_pred_lasso = lasso_regressor.predict(X_test)
import xgboost as xgb
xgb = xgb.XGBRegressor(objective="reg:linear", random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#LINEAR REGRESSION
MSE = mean_squared_error(y_test, y_pred_linreg)
print('Our Linear Regression mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_linreg)
print('Our Linear Regression mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_linreg) 
print('Our Linear Regression R2 score is: ', R2)
print('Our Linear Regreesion Root Mean Squared Error is:', np.sqrt(mean_squared_error(y_test, y_pred_linreg)))
print('-'*100)
print('-'*100)
#RIDGE REGRESSION
MSE = mean_squared_error(y_test, y_pred_ridge)
print('Our Rdige Regression mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_ridge)
print('Our Ridge Regression mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_ridge) 
print('Our Ridge Regression R2 score is: ', R2)
print('Our Ridge Regression Root Mean Squared Error is:', np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
print('-'*100)
print('-'*100)
#LASSO REGRESSION
MSE = mean_squared_error(y_test, y_pred_lasso)
print('Our Lasso Regression mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_lasso)
print('Our Lasso Regression mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_lasso) 
print('Our Lasso Regression R2 score is: ', R2)
print('Our Lasso Regression Root Mean Squared Error is:', np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print('-'*100)
print('-'*100)
#XGBOOST
MSE = mean_squared_error(y_test, y_pred_xgb)
print('Our XGBoost mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_xgb)
print('Our XGBoost mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_xgb) 
print('Our XGBoost R2 score is: ', R2)
print('Our XGBoost Root Mean Squared Error is:', np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
#Cecking Multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
pd.DataFrame({'vif': vif[0:]}, index=X_train.columns).T
#Checking Normality
residual = y_test - y_pred_xgb
sns.distplot(residual)
#Checking Normality
import scipy as sp
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residual, plot=ax, fit=True)
#Checking Homoscedacity
sns.scatterplot(y_pred_xgb, residual)
plt.hlines(y=0, xmin= 1, xmax=5)
plt.xlabel('Residual')
plt.ylabel('Prediksi')
plt.title('Residual Plot')
# Hyper Parameter Tunning
from sklearn.model_selection import RandomizedSearchCV

params={
 "learning_rate"    : [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 0.9, 1.0 ]
    
}
#Using Randomized Search CV to look the best parameter
random_search= RandomizedSearchCV(estimator=xgb,
                                param_distributions=params,
                                cv=5, n_iter=50,
                                scoring = 'r2',n_jobs = 4,
                                verbose = 1, 
                                return_train_score = True,
                                random_state=42)
#Train Hyperparameter into our Data
random_search.fit(X_train, y_train)
#Ceck the best estimator
random_search.best_estimator_
#Using the best hyperparameter into our model
import xgboost as xgb
xgb = xgb.XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1.0, gamma=0.2, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.25, max_delta_step=0, max_depth=10,
             min_child_weight=7, monotone_constraints=None,
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:linear', random_state=42, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
xgb.fit(X_train, y_train)
y_pred_xgb_tunning = xgb.predict(X_test)
#Check Metrics after tunning
MSE = mean_squared_error(y_test, y_pred_xgb_tunning)
print('Our XGBoost after tunning mean squared error is: ',MSE)
MAE = mean_absolute_error(y_test, y_pred_xgb_tunning)
print('Our XGBoost after tunning mean absolute error is: ',MAE)
R2 = r2_score(y_test, y_pred_xgb_tunning) 
print('Our XGBoost after tunning R2 score is: ', R2)
print('Our XGBoost after tunning Root Mean Squared Error is:', np.sqrt(mean_squared_error(y_test, y_pred_xgb_tunning)))
#Visualize The Actual Data and our Prediction
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb_tunning})
result.head(20)
#Visualize using scatter plot
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.set(title="XG Boost Tunning", xlabel="Aktual", ylabel="Prediksi")
ax.scatter(y_test, y_pred_xgb_tunning)
ax.plot([0,max(y_test)], [0,max(y_pred_xgb_tunning)], color='r')
fig.show()