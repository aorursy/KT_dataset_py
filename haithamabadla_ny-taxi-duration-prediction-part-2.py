import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.feature_selection import SelectFromModel

from lightgbm import LGBMRegressor as lgbreg
from xgboost import XGBRegressor as xgbreg
pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')
%matplotlib inline
def import_df(path):
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis = 1, inplace = True)
    return df

df = import_df('../input/training_df.csv')
# Shuffle Dataset
df = shuffle(df)
df.head()
days = ['Morning', 'Noon', 'Afternoon']
df['day_night'] = df.day_part.apply(lambda x: 'Day' if x in days else 'Night')
df.columns
drop_cols = ['year_pick', 'month_pick', 'day_pick', 'weekday_pick', 'hour_pick', 'day_part', 'weather_condition', 'temperature', 'rain', 'season', 'store_and_fwd_flag']
dummies_cols = ['weekdays_weekends', 'taxi_type', 'day_night', 'rush_hours', 'vendor_id', 'passenger_count', 'pickup_area', 'dropoff_area', 'airport_pickup', 'airport_dropoff', 'snow']
target = ['trip_duration']
min_max_cols = ['pickup_latitude_round3', 'pickup_longitude_round3', 'dropoff_latitude_round3', 'dropoff_longitude_round3', 'est_distance', 'avg_speed', 'trip_duration']
# Scaling values
scaler = MinMaxScaler()
df[min_max_cols] = scaler.fit_transform(df[min_max_cols])
# Function for np.log
'''
def log_scale_features(features):
    return round(np.log(features + 1), 2) # Added 1 to avoid infinity numbers

# Function for MinMaxScaler
def min_max_scale_features(features):
    try:
        scaler = MinMaxScaler()
        return scaler.fit_transform(features)
    
    except ValueError as err:
        print(err)
'''

# Function for processing data
def processing_df(data):
    try:
        # Drop columns
        processed_df = data.copy()
        processed_df.drop(drop_cols, axis = 1, inplace = True)

        # Scaling features using MinMaxScaler
        #scaler = MinMaxScaler()
        #processing_df[min_max_cols] = scaler.fit_transform(processing_df[min_max_cols])
        
        # Preparing X
        X = processed_df.drop(target, axis = 1)
        
        # Logarithm features using np.log
        #for feature in log_scale_cols:
        #    X[feature] = log_scale_features(X[feature])
            
        # Creating dummy variables
        X = pd.get_dummies(X, columns= dummies_cols, drop_first= True)
        
        # Preparing y
        #y = log_scale_features(processed_df[target])
        y = np.ravel(processed_df[target]) # To ensure the shape is correct.
        
        feature_names = X.columns

        # Return X, y as arrays
        return X.values, y, feature_names
    
    except ValueError as err:
        print(err)
X, y, feature_names = processing_df(data= df)#, target= 'trip_duration')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 120)
# Training model for feature selection take a very long time, due to that I have decided to reduce the testing size sample to 50000 instead of 993871
testSize = 50000
# Option 1 - Feature selection using Lasso

lassoAlpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.4, 0.8, 1]

def lasso_featuresSelection(X_data, y_data):
    
    plt.figure(figsize = (20, 8))
        
    for i in lassoAlpha:
        
        lasso = Lasso(alpha= i)
        lasso_coef = lasso.fit(X_data,y_data).coef_

        _ = plt.plot(range(len(feature_names)), lasso_coef, label = 'Features COEF with Alpha = {}'.format(i))

    _ = plt.xticks(range(len(feature_names)), feature_names, rotation = 90)
    _ = plt.ylabel('Coefficients')

    plt.title('Lasso - Features COEFon 50,000 samples', pad = 15, weight = 'bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Calling lasso_featuresSelection 
lasso_featuresSelection(X_train[:testSize], y_train[:testSize])
# Option 3 - Feature selection using XGBoostRegressor 

nEstimators = [500, 1000, 2000, 5000]

def xgbr_featuresSelection(X_data,y_data):

    plt.figure(figsize = (20, 8))
        
    for i in nEstimators:
        
        xgbr = xgbreg(n_estimators= i, n_jobs= -1)
        xgbr_featureImportances = xgbr.fit(X_data,y_data).feature_importances_

        _ = plt.plot(range(len(feature_names)), xgbr_featureImportances, label = 'Features Importances with No. of Estimator = {}'.format(i))

    _ = plt.xticks(range(len(feature_names)), feature_names, rotation = 90)
    _ = plt.ylabel('Coefficients')

    plt.title('XGBoost Regressor  - Features Importances', pad = 15, weight = 'bold')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
# Calling rf_featuresSelection 
xgbr_featuresSelection(X_train[:testSize], y_train[:testSize])
# Option 4 - Feature selection using LightGBRegressor 

nEstimators = [500, 1000, 2000, 5000]

def lightgb_featuresSelection(X_data,y_data):

    plt.figure(figsize = (20, 8))
        
    for i in nEstimators:
        
        lgbr = lgbreg(n_estimators= i, n_jobs= -1)
        lgbr_featureImportances = lgbr.fit(X_data,y_data).feature_importances_

        _ = plt.plot(range(len(feature_names)), lgbr_featureImportances, label = 'Features Importances with No. of Estimator = {}'.format(i))

    _ = plt.xticks(range(len(feature_names)), feature_names, rotation = 90)
    _ = plt.ylabel('Coefficients')

    plt.title('LightGB Regressor - Features Importances', pad = 15, weight = 'bold')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
# Calling rf_featuresSelection 
lightgb_featuresSelection(X_train[:testSize], y_train[:testSize])
# Preparing the baseline to work againt

avg_tripDuration = round(np.mean(y_train),2)
baseline_pred = np.repeat(avg_tripDuration, y_test.shape[0])
baseline_rmse = np.sqrt(mean_squared_error(baseline_pred, y_test))

print("Basline RMSE of Validation data :",baseline_rmse)
# Just for the purpose of reducing the proccessing time 
size_Xy_train = 35000 
size_Xy_test = 8000

modelList = []
training_score_list = []
testing_score_list = []
rmse_list = []
train_rmse_list = []
variance_list = []

# Creating models dictionary
models = {'Linear Regression': LinearRegression(n_jobs= -1), 
          'Random Forest Regressor': RandomForestRegressor(n_estimators= 2000, n_jobs= -1),
          'XGBoost Regressor': xgbreg(n_estimators= 2000, n_jobs= -1),
          'LightGBM Regressor': lgbreg(n_estimators= 2000, n_jobs= -1)}

# Selecting model function
def selecting_model():
    
    X_train_reduced = X_train[:size_Xy_train]
    y_train_reduced = y_train[:size_Xy_train]
    
    X_test_reduced = X_test[:size_Xy_test] 
    y_test_reduced = y_test[:size_Xy_test]
    
    for model_name, model in models.items():
        
        regressor = model
        regressor.fit(X_train_reduced, y_train_reduced)
        y_pred = np.round(regressor.predict(X_test_reduced), 2)

        score_trainingSet = regressor.score(X_train_reduced, y_train_reduced)
        score_testingSet = regressor.score(X_test_reduced, y_test_reduced)
        
        rmse = np.sqrt(mean_squared_error(y_pred, y_test_reduced))
        train_rmse = np.sqrt(mean_squared_error(regressor.predict(X_train_reduced), y_train_reduced))
        variance = abs(train_rmse - rmse)
        
        modelList.append(model_name)
        training_score_list.append(score_trainingSet)
        testing_score_list.append(score_testingSet)
        rmse_list.append(rmse)
        train_rmse_list.append(train_rmse)
        variance_list.append(variance)
        
        print("Model: {}".format(model_name))
        print("Score on Training Dataset is: {}".format(score_trainingSet))
        print("Score on Testing Dataset is: {}".format(score_testingSet))
        print("Test RMSE is: {}".format(rmse))
        print("Train RMSE is: {}".format(train_rmse))
        print("Variance is: {}\n".format(variance))
        
# Call selecting_model function
selecting_model()
# Create dataframe with the generated results
results = {'model_name': modelList, 'training_score_list': training_score_list, 'testing_score_list': testing_score_list,'rmse_test': rmse_list, 'rmse_train': train_rmse_list, 'varience': variance_list}
results_df = pd.DataFrame(results)
results_df
# Plot results

fig, ax = plt.subplots(ncols=5, figsize= (40,8))

sns.barplot(x = 'model_name', y = 'training_score_list', data = results_df, ax= ax[0])
sns.barplot(x = 'model_name', y = 'testing_score_list', data = results_df, ax= ax[1])
sns.barplot(x = 'model_name', y = 'rmse_train', data = results_df, ax= ax[2])
sns.barplot(x = 'model_name', y = 'rmse_test', data = results_df, ax= ax[3])
sns.barplot(x = 'model_name', y = 'varience', data = results_df, ax= ax[4])

ax[0].set_title('Accuracy Score on Training Dataset', pad = 15, weight = 'bold')
ax[0].set_xlabel('Model')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_ylabel('Accuracy Score')

ax[1].set_title('Accuracy Score on Testing Dataset', pad = 15, weight = 'bold')
ax[1].set_xlabel('Model')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
ax[1].set_ylabel('Accuracy Score')

ax[2].set_title('RMSE Score on Training Dataset', pad = 15, weight = 'bold')
ax[2].set_xlabel('Model')
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90)
ax[2].set_ylabel('RMSE Score')

ax[3].set_title('RMSE Score on Testing Dataset', pad = 15, weight = 'bold')
ax[3].set_xlabel('Model')
ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=90)
ax[3].set_ylabel('RMSE Score')

ax[4].set_title('Testing and Training Varience', pad = 15, weight = 'bold')
ax[4].set_xlabel('Model')
ax[4].set_xticklabels(ax[3].get_xticklabels(), rotation=90)
ax[4].set_ylabel('Varience')

plt.show()

# Reducing number os samples
sample_size = 10000 

xgb = xgbreg(gamma= 0.1, max_depth= 30)

# 'objective' is 'reg:linear' by default for loss function to be minimized
# 'nthread' = -1 by default to use all processors 
# eval_metric = rmse by default for XGBoost Regressor
# `eta` is alias name for 'learning_rate'

params = {'learning_rate': [0.07, 0.1],
          'min_child_weight': [5, 10],
          'reg_alpha': [0, 0.005, 0.008], # Default is 0
          'silent': [1], # Silent mode is activated is set to 1, i.e. no running messages will be printed.
          'subsample': [0.6, 0.9],
          'colsample_bytree': [0.6, 0.9],
          'n_estimators': [300, 1000, 3000]}

xgb_grid = RandomizedSearchCV(xgb, params, cv = 5)

xgb_grid.fit(X[:sample_size], y[:sample_size])

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
xgb_results = pd.DataFrame(xgb_grid.cv_results_).sort_values('mean_test_score', ascending=False)
xgb_results[:5]