# General
import warnings
from datetime import datetime
import os

# Data Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Linear Regression
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
import statsmodels.formula.api as smf

# Neural Networks
from keras.models import Model
from keras import Input, layers

# Random Forests
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# XGBoost
import xgboost as xg
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

%matplotlib inline 
#to show images within Jupyter Notebook without having to call "show()"
# Load Datasets from Computer
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')
# sample_submission = pd.read_csv('sampleSubmission.csv')

# Load Datasets from Kaggle Kernel
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sampleSubmission.csv")

# Visualize Datasets as Tables
display(train_df.head(3))
print('Training Set Size: ', train_df.shape)
display(test_df.head(3))
print('Test Set Size: ', test_df.shape)
display(sample_submission.head(3))
print(train_df.info())
undesired_feat1 = ['casual', 'registered', 'atemp']
train_df.drop(undesired_feat1, inplace=True, axis=1)
sns.set()
sns.pairplot(train_df, hue='weather')
warnings.simplefilter('ignore')
# Change Datetime to date and time values and make datetime the axis
train_df['datetime'] = train_df['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

# Add columns for month, hour, day and year
train_df['year'] = train_df.datetime.apply(lambda x: x.year)
train_df['month'] = train_df.datetime.apply(lambda x: x.month)
train_df['day'] = train_df.datetime.apply(lambda x: x.day)
train_df['hour'] = train_df.datetime.apply(lambda x: x.hour)
train_df['weekday'] = train_df.datetime.apply(lambda x: x.weekday())

# Set datetime as the index
train_df.set_index('datetime', inplace=True)

train_df.head(2)
fig,ax = plt.subplots()
fig.set_size_inches(12,5)
sns.scatterplot(x = train_df.index, y = train_df['count'].rolling(24).sum())
ax.set(ylabel='Count Rolling Sum', xlabel='Time', title='Ridership Rolling Sum Over 24 Hours')
ax.set_xlim(train_df.index.min(), train_df.index.max())
fig, axes = plt.subplots(nrows=3,ncols=2)
fig.set_size_inches(14, 15)

# As a function of time of day
sns.boxplot('temp', 'count', data=train_df, ax=axes[0][0])
axes[0][0].set(ylabel='Count', xlabel='Temp', title='Count vs Temperature')
for item in axes[0][0].get_xticklabels():
    item.set_rotation(90)
    
# As a function of month
sns.boxplot('month', 'count', data=train_df, ax=axes[0][1])
axes[0][1].set(ylabel='Count', xlabel='Month', title='Count vs Month')

# As a function of hour in the day
sns.boxplot('hour', 'count', data=train_df, ax=axes[1][0])
axes[1][0].set(ylabel='Count', xlabel='Time of Day', title='Count vs Time-of-Day')

# As a function of the day of the month
sum_count_per_hour = pd.DataFrame(train_df.groupby('hour')['count'].sum()).reset_index()
sns.lineplot(x='hour', y='count', data=sum_count_per_hour, ax=axes[1][1])
axes[1][1].set(ylabel='Sum of Count', xlabel='Time of Day', title='Count vs Sum of Count Per Time of Day')

# As a function of day of the week
sns.boxplot(x='weekday', y='count', data=train_df, ax=axes[2][0])
axes[2][0].set(ylabel='Count', xlabel='Weekday', title='Count vs Weather')

# As a function of weather
sns.boxplot('weather', 'count', data=train_df, ax=axes[2][1])
axes[2][1].set(ylabel='Count', xlabel='Weather', title='Count vs Weather')

plt.tight_layout() # to leave some space between graphs
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(14, 5)

sns.distplot(train_df['count'], ax=axes[0])
axes[0].set(title='Original Distribution')

# Define Data Variations
log_count = np.log(train_df['count'])
x_p5 = np.sqrt(train_df['count'])

sns.distplot(log_count, ax=axes[1])
axes[1].set(title='Modified Distribution')
# Keep both log_count and count in dataset to be able to go back and forth for intepretability
train_df['log_count'] = np.log(train_df['count']) 

# Remove Outliers
count_mean = np.mean(train_df['count'])
count_std = np.std(train_df['count'])
three_std = count_mean + 3*count_std

train_df = train_df[train_df['count'] < three_std]
mask = np.array(train_df.corr())
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(17,10)
sns.heatmap(train_df.corr(), mask=mask, square=True,annot=True)
fig,axes = plt.subplots(nrows=3, ncols=1)
fig.set_size_inches(15, 18)

# Question 1:
# 0=Monday, 6=Sunday
day_of_week = train_df.groupby(['weekday', 'hour'], sort=True).mean().reset_index()
sns.pointplot(x='hour', y='count', data=day_of_week, hue='weekday', join=True, ax=axes[0], scale=1.5, palette='husl')
axes[0].set(xlabel='Time of the Day', ylabel='Count', title='Count vs Hour, Separated by Weather Type')

# Question 2:
yearly_workingday = train_df.groupby(['workingday', 'month'], sort=True).mean().reset_index()
sns.pointplot(x='month', y='count', data=yearly_workingday, hue='workingday', join=True, ax=axes[1], scale=1.5, palette='husl')
axes[1].set(xlabel='Month', ylabel='Count', title='Count vs Month, Separated by Work vs. Weekend Day')

# Question 3: 
# cat_temp = pd.cut(train_df['temp'], 4, labels=["cold", "cool", "warm", "hot"])
# train_df['cat_temp'] = cat_temp
hourly_temp = train_df.groupby(['weather', 'hour'], sort=True).mean().reset_index()
sns.pointplot(x='hour', y='count', data=train_df, hue='weather', join=True, ax=axes[2], scale=1.5, palette='husl')
axes[2].set(xlabel='Time of the Day', ylabel='Count', title='Count vs Hour, Separated by Temp Type')

plt.tight_layout() # to leave some space between graphs
def load_datasets(file_names):
    '''loads a list of files as pandas dataframes'''
    files_list = []
    for i in file_names:
        df_i = pd.read_csv("../input/{}".format(i))
        files_list.append(df_i)
    return(files_list)

def drop_features(df, features):
    '''Drop specified list of features'''
    df.drop(features, inplace=True, axis=1)
    return(df)

def process_datetime(df):
    '''Change Datetime to date and time values, set those values as columns, and
    set original datetime column as index'''
    df['datetime'] = df['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    # Add columns for month, hour, day and year
    df['year'] = df.datetime.apply(lambda x: x.year)
    df['month'] = df.datetime.apply(lambda x: x.month)
    df['day'] = df.datetime.apply(lambda x: x.day)
    df['hour'] = df.datetime.apply(lambda x: x.hour)
    df['weekday'] = df.datetime.apply(lambda x: x.weekday())
    df.set_index('datetime', inplace=True)
    return(df)

def log_transform(df):
    '''Keep both log_count and count in dataset to be able to go back and 
    forth for intepretability'''
    df['count'] = np.log(df['count']) 
    return(df)

def remove_outliers(df):
    '''Remove Outliers'''
    count_mean = np.mean(df['count'])
    count_std = np.std(df['count'])
    three_std = count_mean + 3*count_std
    df = df[train_df['count'] < three_std]
    return(df)

def norm_scale(df, cont_features):
    '''Normalize and Scale Continuous Features'''
    df[cont_features] -= np.mean(df[cont_features])
    df[cont_features] /= np.std(df[cont_features])
    return(df)

def cat_into_one_hot(df, cat_features):
    '''Takes list of categorical features and turns them into One-Hot Encoding'''
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)
    return(df)

def split_train_val(df, val_size):
    '''Separates the training set into a training and a validation set
    val_size should be a number between 0 and 1'''
    split_at = int(df.shape[0] // (1/(1-val_size)))
    df_train = df[:split_at][:]
    df_val = df[split_at:][:]
    return(df_train, df_val)
    
def split_x_y(df_train, df_val):
    '''Takes a training set and a validation set and returns X_train, y_train, X_val, y_val'''
    x_train_df = df_train.drop('count', axis=1)
    y_train_df = df_train['count']
    x_val_df = df_val.drop('count', axis=1)
    y_val_df = df_val['count']
    return(x_train_df, x_val_df, y_train_df, y_val_df)

def exp_transform(df):
    '''To convert Count back to non-log values'''
    if type(df) == pd.core.series.Series:
        df = np.exp(df)
    elif type(df) == np.ndarray:
        df = np.exp(df)
    else:
        df['count'] = np.exp(df['count'])
    return(df)
# Load Datasets
train_df, test_df = load_datasets(['train.csv', 'test.csv'])

# Drop Uninportant Features
train_df = drop_features(train_df, ['casual', 'registered', 'atemp'])
test_df = drop_features(test_df, 'atemp')

# Process Datetime
train_df = process_datetime(train_df)
test_df = process_datetime(test_df)

# # Remove Outliers - Only for Training Data
# train_df = remove_outliers(train_df)

# Normalize and Standardize Continuous Variables
cont_features = ['temp', 'humidity', 'windspeed']
train_df = norm_scale(train_df, cont_features)
test_df = norm_scale(test_df, cont_features)

# Transform Data to Log - Only for Training Data
train_df = log_transform(train_df)

# Change Categorical into One_Hot - 'day' is not included bc it's different between train & test
cat_features = ['season', 'holiday', 'workingday', 'weather', 'month', 'hour', 'weekday']
train_df = cat_into_one_hot(train_df, cat_features)
test_df = cat_into_one_hot(test_df, cat_features)

# Split into train train and validation sets
train_train_df, train_val_df = split_train_val(train_df, 0.2)

# Split into X and y
X_train, X_val, y_train, y_val = split_x_y(train_train_df, train_val_df)

# Check datasets shapes
print('We check the datasets shapes to ensure our pre-processing function did its job correctly: ', '\n')
print('X_train Shape: ', X_train.shape)
print('X_val Shape: ', X_val.shape)
print('y_train Shape: ', y_train.shape)
print('y_val Shape: ', y_val.shape)
print('Test Set Shape: ', test_df.shape)
X_train_lr = X_train.copy()
y_train_lr = y_train.copy()
X_val_lr = X_val.copy()
y_val_lr = y_val.copy()

feats_to_drop = ['weather_4', 'month_4', 'weekday_1', 'weekday_2']
X_train_lr.drop(feats_to_drop, inplace=True, axis=1)
X_val_lr.drop(feats_to_drop, inplace=True, axis=1)

X_train_lr_intercept = sm.add_constant(X_train_lr, has_constant='add') # this adds an intercept term column
est = sm.OLS(y_train_lr, X_train_lr_intercept) 
est2 = est.fit()
X_val_lr_intercept = sm.add_constant(X_val_lr, has_constant='add')
pred_lr = est2.predict(X_val_lr_intercept)
print(est2.summary())

# Convert back from Log_Count to Count
pred_lr = exp_transform(pred_lr)
y_val_lr = exp_transform(y_val_lr)

# Evaluate Model
pred_lr[pred_lr<0] = 0
msle = np.sqrt(mean_squared_log_error(y_val_lr, pred_lr))
print('MSLE: ', msle)
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(pred_lr, ax=axes[0])
axes[0].set(xlabel='Predicted Count', ylabel='P(count)', title='Val Set Predictions Dist' )

residuals = (y_val_lr - pred_lr)
sns.scatterplot(x=pred_lr, y=residuals, ax=axes[1])
axes[1].set(xlabel='Predicted Count', ylabel='Residuals', title='Residuals vs Predictions')

print('LINEAR REGRESSION WITH LOG TRANSFORMATION ON DATA')
print('MSLE: ', msle)
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(pred_lr, ax=axes[0])
axes[0].set(xlabel='Predicted Count', ylabel='P(count)', title='Val Set Predictions Dist' )

residuals = (y_val_lr - pred_lr)
sns.scatterplot(x=pred_lr, y=residuals, ax=axes[1])
axes[1].set(xlabel='Predicted Count', ylabel='Residuals', title='Residuals vs Predictions')

print('LINEAR REGRESSION WITHOUT LOG TRANSFORMATION ON DATA')
print('MSLE:', msle)
# # Build Network
input_tensor = Input(shape=(X_train.shape[1],))
x = layers.Dense(80, activation='relu')(input_tensor)
x = layers.Dense(80, activation='relu')(x)
output_tensor = layers.Dense(1)(x)
nn_model = Model(input_tensor, output_tensor)
nn_model.summary()

# Compile and Fit
nn_model.compile(optimizer='rmsprop', loss='msle')
history = nn_model.fit(X_train, y_train, validation_split=0.2, epochs=10)
pred_nn = nn_model.predict(X_val)

# # Convert back from Log_Count to Count
# pred = exp_transform(pred)
# y_val = exp_transform(y_val)
# Evaluate Model
pred_nn = np.ravel(pred_nn)
pred_nn[pred_nn<0] = 0
msle = np.sqrt(mean_squared_log_error(y_val, pred_nn))

# Plot Training and Validation MSLE
msle_train = history.history['loss']
msle_val = history.history['val_loss']
epochs = range(1, len(msle_train) + 1)
plt.figure()
plt.plot(epochs, msle_train, 'bo', label='Training msle')
plt.plot(epochs, msle_val, 'b', label='Validation msle')
plt.title('Training and Validation Loss')
plt.legend()

# Plot Predictions and Residuals
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(15, 5)

sns.distplot(pred_nn, ax=axes[0])
axes[0].set(xlabel='Predicted Count', ylabel='P(count)', title='Val Set Predictions Dist' )

residuals = (y_val - pred_nn)
sns.scatterplot(x=pred_nn, y=residuals, ax=axes[1])
axes[1].set(xlabel='Predicted Count', ylabel='Residuals', title='Residuals vs Predictions')

print('NEURAL NETWORKS MODEL RESULTS')
print('MSLE: ', msle)
rfc = RandomForestRegressor(n_estimators = 50)
rfc.fit(X_train, y_train)
rfc.feature_importances_
pred_rf = rfc.predict(X_val)

# Evaluate Model on Validation Set
pred_rf.resize(y_val.shape)
mlse_rf = np.sqrt(mean_squared_log_error(y_val, pred_rf))
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(pred_rf, ax=axes[0])
axes[0].set(xlabel='Predicted Count', ylabel='P(count)', title='Val Set Predictions Dist' )

residuals = (y_val - pred_rf)
sns.scatterplot(x=pred_rf, y=residuals, ax=axes[1])
axes[1].set(xlabel='Predicted Count', ylabel='Residuals', title='Residuals vs Predictions')

print('RANDOM FORESTS MODEL RESULTS')
print('MSLE: ', msle)
xgr_params = [{'max_depth': [8], 'min_child_weight':[4], 'gamma':[0.0001]}] #per previous run

xgr = xg.XGBRegressor()

grid_xgr = GridSearchCV(xgr, param_grid=xgr_params, cv=5, refit=True, verbose=1)
grid_xgr.fit(X_train, y_train)
pred_xgr = grid_xgr.predict(X_val)

best_score = grid_xgr.best_score_
best_params = grid_xgr.best_params_
# Evaluate Model on Validation Set
pred_xgr[pred_xgr<0] = 0
msle_xgr = np.sqrt(mean_squared_log_error(y_val, pred_xgr))

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(pred_xgr, ax=axes[0])
axes[0].set(xlabel='Predicted Count', ylabel='P(count)', title='Val Set Predictions Dist' )

residuals = (y_val - pred_xgr)
sns.scatterplot(x=pred_xgr, y=residuals, ax=axes[1])
axes[1].set(xlabel='Predicted Count', ylabel='Residuals', title='Residuals vs Predictions')

print('XGBOOST REGRESSOR WITHOUT LOG TRANSFORMATION MODEL RESULTS')
print('MSLE: ', msle_xgr)
egb_params = [{'learning_rate': [0.1], 'max_depth':[6]}] # from previous run

egb = GradientBoostingRegressor(n_estimators=50, loss='ls')

grid_egb = GridSearchCV(egb, param_grid=egb_params, cv=5, refit=True, verbose=1)
grid_egb.fit(X_train, y_train)
best_score = grid_egb.best_score_
best_params = grid_egb.best_params_
print(' The Best Score :', best_score, '\n', 'The Best Params : ', best_params)
pred_egb = grid_egb.predict(X_val)
pred_egb[pred_egb<0] = 0

# Evaluate Model on Validation Set
# pred.resize(y_val_xg.shape)
msle_egb = np.sqrt(mean_squared_log_error(y_val, pred_egb))
print('MSLE Score: ', msle_egb)
# Evaluate Model on Validation Set
pred_egb[pred_egb<0] = 0
msle_egb = np.sqrt(mean_squared_log_error(y_val, pred_egb))

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 5)
sns.distplot(pred_egb, ax=axes[0])
axes[0].set(xlabel='Predicted Count', ylabel='P(count)', title='Val Set Predictions Dist' )

residuals = (y_val - pred_egb)
sns.scatterplot(x=pred_egb, y=residuals, ax=axes[1])
axes[1].set(xlabel='Predicted Count', ylabel='Residuals', title='Residuals vs Predictions')

print('ENSEMBLE GRADIENT BOOSTING WITH LOG TRANSFORMATION MODEL RESULTS')
print('MSLE: ', msle_egb)
# Concatenate train and validation sets into one
X_train_all = pd.concat([X_train, X_val], axis=0)
y_train_all = pd.concat([y_train, y_val], axis=0)
X_train_lr_all = sm.add_constant(X_train_all, has_constant='add')
test_df_lr = sm.add_constant(test_df, has_constant='add')
est = sm.OLS(y_train_all, X_train_lr_all) 
est2 = est.fit()
pred_lr = est2.predict(test_df_lr)

# Convert back from Log_Count to Count
pred_lr = exp_transform(pred_lr)
pred_lr[pred_lr<0] = 0
egb_params = [{'learning_rate': [0.1], 'max_depth':[6]}] # from previous run

egb = GradientBoostingRegressor(n_estimators=250, loss='ls')
grid_egb = GridSearchCV(egb, param_grid=egb_params, verbose=1)
grid_egb.fit(X_train_all, y_train_all)

pred_egb = grid_egb.predict(test_df)
pred_egb = exp_transform(pred_egb)
pred_egb[pred_egb<0] = 0
# # Make a copy of the sample submission file for each model
# submission_LR = sample_submission.copy()
# submission_EGB = sample_submission.copy()

# # Add predictions to submission file
# submission_LR['count'] = np.array(pred_lr)
# submission_EGB['count'] = pred_egb

# # Save Files
# submission_LR.to_csv('submission_LR', header=True, index=False)
# submission_EGB.to_csv('submission_EGB', header=True, index=False)