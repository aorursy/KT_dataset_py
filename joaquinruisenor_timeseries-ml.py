import pandas as pd
import numpy as np
import datetime
%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
%matplotlib inline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

#ML imports
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
df_gb = pd.read_csv('../input/timeseries-prophet/Energy_PJM.csv')
df_gb.head()
def create_features(df_split, label=None): 
    df_split = df_split.reset_index()
    df_split['Datetime'] =  pd.to_datetime(df_split['Datetime'], infer_datetime_format=True)
    df_split['hour'] = df_split['Datetime'].dt.hour
    df_split['dayofweek'] = df_split['Datetime'].dt.dayofweek
    df_split['quarter'] = df_split['Datetime'].dt.quarter
    df_split['month'] = df_split['Datetime'].dt.month
    df_split['year'] = df_split['Datetime'].dt.year
    df_split['dayofyear'] = df_split['Datetime'].dt.dayofyear
    df_split['dayofmonth'] = df_split['Datetime'].dt.day
    df_split['weekofyear'] = df_split['Datetime'].dt.weekofyear
    
    X = df_split[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df_split[label]
        return X, y
    X = X.drop(label, axis=1, inplace=True)
    return X
#We split the data into training and testing set
df_gb.set_index('Datetime', inplace = True)
split_date = '2017-01-01'
df_train = df_gb.loc[df_gb.index <= split_date].copy()
df_test = df_gb.loc[df_gb.index > split_date].copy()
x_train, y_train = create_features(df_train, label='Energy')
x_test, y_test = create_features(df_test, label='Energy')
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=50,
       verbose=True)
_ = plot_importance(reg, height=0.9)
df_test['Energy Prediction'] = reg.predict(x_test) #this predicts the values of energy using de X_test data and puts
#it into a new column in df_test
predictions = [round(value) for value in df_test['Energy Prediction']]
df_all = pd.concat([df_test, df_train], sort=False)
_ = df_all[['Energy','Energy Prediction']].plot(figsize=(15, 5))
y_prediction_on_train = reg.predict(x_train)
y_prediction_on_test = reg.predict(x_test)
MSE_Train = mean_squared_error(y_train,y_prediction_on_train )
MSE_Test = mean_squared_error(y_test,y_prediction_on_test )
RMSE_Train = sqrt(MSE_Train)
RMSE_Test = sqrt(MSE_Test)
print('RMSE_Train: '+ str(RMSE_Train))
print('RMSE_Test: '+ str(RMSE_Test))
## Hyper Parameter Optimization
base_score=[0.25,0.5,0.75,1]
n_estimators = [900, 1100, 1500] #the number of decision trees it can have
max_depth = [2, 3, 5] #the nr of depthd of the trees
booster=['gbtree'] #probably will choose gbtree but we give it the choice either way to choose a linear regression
learning_rate=[0.05,0.1,0.15,0.20] #values for different learning rates
min_child_weight=[2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }

#Remember all these hyperparameters are used inside XGBoost Regressor, othr wise it wont be able to use them, see with
#the Shift+tab
# Set up the random search with 4-fold cross validation usinv CV=5
random_cv = RandomizedSearchCV(estimator=reg,
            param_distributions=hyperparameter_grid, #we use the grid we defined before
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True, #getting the training score
            random_state=42)
random_cv.fit(x_train,y_train) 
random_cv.best_estimator_
import xgboost
from numpy import nan
regressor=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.2, max_delta_step=0, max_depth=3,
             min_child_weight=2, missing=nan, monotone_constraints=None,
             n_estimators=1100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
#inserted values into the initialization of the Algo

regressor.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=50,
       verbose=True)
df_test['Energy Prediction Regularized'] = regressor.predict(x_test) #this predicts the values of energy using de X_test data and puts
#it into a new column in df_test
predictions = [round(value) for value in df_test['Energy Prediction Regularized']]
df_all_reg = pd.concat([df_test, df_train], sort=False)
pl = df_all_reg[['Energy','Energy Prediction Regularized']].plot(figsize=(15, 5))
t = plot_importance(regressor, height=0.9)
y_prediction_on_train_reg = regressor.predict(x_train)
y_prediction_on_test_reg = regressor.predict(x_test)
MSE_Train_reg = mean_squared_error(y_train,y_prediction_on_train_reg )
MSE_Test_reg = mean_squared_error(y_test,y_prediction_on_test_reg )
RMSE_Train_reg = sqrt(MSE_Train_reg)
RMSE_Test_reg = sqrt(MSE_Test_reg)
print('RMSE_Train: '+ str(RMSE_Train_reg))
print('RMSE_Test: '+ str(RMSE_Test_reg))
import xgboost
from numpy import nan
prueba=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints=None,
             learning_rate=0.5, max_delta_step=0, max_depth=5,
             min_child_weight=2, missing=nan, monotone_constraints=None,
             n_estimators=1100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
             validate_parameters=False, verbosity=None)
#inserted values into the initialization of the Algo

prueba.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=50,
       verbose=True)
df_test.head()
df_test.dtypes
c = df_test.reset_index()
c['Datetime'] =  pd.to_datetime(c['Datetime'], infer_datetime_format=True)
c[["Datetime", "Energy", "Energy Prediction"]].plot(x="Datetime", kind="line")
plt.xlim(datetime.date(2017,1,1), datetime.date(2018,1,1))

c = df_test.reset_index()
c['Datetime'] =  pd.to_datetime(c['Datetime'], infer_datetime_format=True)
c[["Datetime", "Energy", "Energy Prediction"]].plot(x="Datetime", kind="line")
plt.xlim(datetime.date(2017,12,1), datetime.date(2017,12,31))