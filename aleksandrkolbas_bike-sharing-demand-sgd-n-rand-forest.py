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
from sklearn import model_selection, linear_model, ensemble, pipeline, preprocessing, metrics

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Reading the train data and test data
raw_data = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
raw_val = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
raw_data.head()
raw_val.head()
print(raw_data.shape, raw_val.shape)
raw_data.isnull().sum()
raw_val.isnull().sum()
raw_data.info()
raw_data.describe()
# Correlation 
raw_data.corr()
# Visualusation of correlation 
corr = raw_data.corr()
plt.figure(figsize=(11, 9))
ax = sns.heatmap(corr, square=True,annot=True,cbar=True, linewidths=.5)
i, k = ax.get_ylim()
ax.set_ylim(i+0.5, k-0.5)
# Features 'registered', 'casual' and target variable 'count'
# have a strong linear dependence (a+b=c)
np.all(raw_data.registered + raw_data.casual == raw_data['count'])
# Converting column to datetime type 
raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)
raw_val.datetime = raw_val.datetime.apply(pd.to_datetime)
# Adding new columns - month, hour and year
raw_data['month'] = raw_data.datetime.apply(lambda x : x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x : x.hour)
raw_data['year'] = raw_data.datetime.apply(lambda x : x.year)

raw_val['month'] = raw_val.datetime.apply(lambda x : x.month)
raw_val['hour'] = raw_val.datetime.apply(lambda x : x.hour)
raw_val['year'] = raw_val.datetime.apply(lambda x : x.year)
# Delelting of ununnecessary features
raw_data = raw_data.drop(['datetime', 'casual', 'registered'], axis=1)
val_data = raw_val.drop(['datetime'], axis=1)
# Dividing to train sample and label sample
data_labels = raw_data['count']
data = raw_data.drop(['count'], axis=1)
# Spliting data into random train and test subsets
X_train, X_test, y_train, y_test = model_selection.train_test_split(data,
                                                                    data_labels,
                                                                    test_size=0.1,
                                                                    random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Noticing colums with different type of features
binary_col = ['holiday', 'workingday']
numeric_col = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
categor_col = ['season', 'weather', 'month', 'year']

# Function for getting index of columns (for FunctionTransformer)
def get_ind(names_col):
    return [X_train.columns.get_loc(i) for i in names_col]
# Choosing a model
sgd_reg = linear_model.SGDRegressor(random_state=42)
# Creating pipeline for data transforming and further estimation
sgd_estimator = pipeline.Pipeline(steps=[
    ('feature_processing', pipeline.FeatureUnion(transformer_list=[
        
        #binary
        ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data.iloc[:, get_ind(binary_col)])),
        
        #numeric
        ('numeric_variables_processing', pipeline.Pipeline(steps=[
            ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, get_ind(numeric_col)])),
            ('scaling', preprocessing.StandardScaler(with_mean=0.))])),
        
        #categorical
        ('categorical_variables_processing', pipeline.Pipeline(steps=[
            ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, get_ind(categor_col)])),
            ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore'))])),
        ])),
    
    ('model_fitting', sgd_reg)
])
# Available parametrs of created estimator
print(*sgd_estimator.get_params().keys(), sep='\n')
# Variation of paramers for grid_search
parametrs_sgd = {'model_fitting__alpha' : [0.00001, 0.0001, 0.001, 0.01, 0.1],
                 'model_fitting__eta0' : [0.001, 0.05, 0.1, 0.5], # initial gradient step
                 'model_fitting__max_iter' : [500, 1000, 2000],
                 'model_fitting__penalty' : ['l2']}
# Available scores of grid_search
print(*metrics.SCORERS.keys(), sep='\n')
# Choosing grid_search for searching best model's parametrs and cross-validation
grid_cv_sgd = model_selection.GridSearchCV(sgd_estimator,
                                           parametrs_sgd,
                                           scoring='neg_mean_absolute_error',
                                           cv=5,
                                           n_jobs=-1)
# Fiting grid_search by data
grid_cv_sgd.fit(X_train, y_train)
print(grid_cv_sgd.best_score_)
print(grid_cv_sgd.best_params_)
def get_rmsle(y_true, y_pred):
    '''Func for counting Root Mean Squared Logarithmic Error (RMSLE)'''
    
    # scaling to (0. 1)
    y_true_scaled = preprocessing.minmax_scale(y_true,feature_range=(0,1))    
    y_pred_scaled = preprocessing.minmax_scale(y_pred, feature_range=(0,1))
    
    return np.sqrt(metrics.mean_squared_log_error(y_true_scaled,
                                                  y_pred_scaled))
y_train_pred_sgd = grid_cv_sgd.best_estimator_.predict(X_train)
# RMSLE on train data
print('RMSLE on train data: {}'.format(get_rmsle(y_train, y_train_pred_sgd)))
y_test_pred_sgd = grid_cv_sgd.best_estimator_.predict(X_test)
# RMSLE on test data
print('RMSLE on test data: {}'.format(get_rmsle(y_test, y_test_pred_sgd)))
# compare several test labels and predicted labels just by view 
print(y_test[:13].to_numpy())
print(y_test_pred_sgd[:13].round())
# Display a graph of points in the space of correct labels and predictions 
# on train and test data. A diagonal cloud of point is expected for a good regression model.\
# (2 scatter plots on 1 graph)
plt.figure(figsize=(7, 5))
plt.grid(True)
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.scatter(y_train, y_train_pred_sgd, alpha=0.5, color='red', label='train')
plt.scatter(y_test, y_test_pred_sgd, alpha=0.5, color='blue', label='test')
plt.legend()
plt.title('SGD Regressor')
# Choosing a model
rf_reg = ensemble.RandomForestRegressor(random_state=42)
# Creating pipeline for data transforming and further estimation
# (notice: numeric data doesn't need scaling with random forest, but just let it be)
rf_estimator = pipeline.Pipeline(steps=[
    ('feature_processing', pipeline.FeatureUnion(transformer_list=[
        
        #binary
        ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data.iloc[:, get_ind(binary_col)])),
        
        #numeric
        ('numeric_variables_processing', pipeline.Pipeline(steps=[
            ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, get_ind(numeric_col)])),
            ('scaling', preprocessing.StandardScaler(with_mean=0.))])),        
        
        #categorical
        ('categorical_variables_processing', pipeline.Pipeline(steps=[
            ('selecting', preprocessing.FunctionTransformer(lambda data: data.iloc[:, get_ind(categor_col)])),
            ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False))])),
        ])),
    
    ('model_fitting', rf_reg)
])
# Available parametrs of created estimator
print(*rf_estimator.get_params().keys(), sep='\n')
# Variation of paramers for grid_search
rf_parametrs = {'model_fitting__n_estimators' : [500, 1000],
                'model_fitting__max_depth' : [20, 60]}
# Choosing grid_search for searching best model's parametrs and cross-validation
grid_cv_rf = model_selection.GridSearchCV(rf_estimator,
                                          rf_parametrs,
                                          scoring='neg_mean_absolute_error',
                                          cv=5,
                                          n_jobs=-1)
%%time
# Fiting grid_search by data
grid_cv_rf.fit(X_train, y_train)
print(grid_cv_rf.best_score_)
print(grid_cv_rf.best_params_)
# Dirty hack for getting column name of train data after scalling in order to
# counting feature's importances (see below)
temp_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
temp_train_X_encoded = temp_encoder.fit_transform(X_train[categor_col])
temp_column_name = temp_encoder.get_feature_names(categor_col)
temp_X_train_drop = X_train.drop(categor_col, axis=1)
train_column_names = np.concatenate([np.array(list(temp_X_train_drop)), temp_column_name])
# Let's see which features are most important
feature_importances = grid_cv_rf.best_estimator_.named_steps['model_fitting'].feature_importances_

# puting features and their importances into DataFrame
feature_importan_df = pd.DataFrame({'feature' : train_column_names, 
                                    'feature_importances' : feature_importances.round(4)})
# showing importance of features (first ten)
feature_importances_df = feature_importan_df.sort_values('feature_importances',
                                                         ascending=False).head(10)
feature_importances_df
# Another dirty hack for getting all scalling train data with column name in order to
# shoing plot of dependence of the most important features from count (see below)
one_hot_encoded_data = X_train
for cat in categor_col:      
    temp_one_hot_encoded_data = pd.get_dummies(X_train[cat],prefix=cat)    
    one_hot_encoded_data = pd.concat([one_hot_encoded_data,temp_one_hot_encoded_data],axis=1)
    one_hot_encoded_data.drop(cat, axis=1, inplace=True)  
# Plot of dependence of the most important features from count
plt.figure(figsize=(15, 8))
colors = ['green', 'blue', 'red', 'yellow', 'pink', 'grey']
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.grid(True)
    plt.scatter(one_hot_encoded_data[feature_importances_df.iloc[i, 0]],
                y_train, color=colors[i], label='train')    
    plt.ylabel('count')
    plt.xlabel(feature_importances_df.iloc[i, 0])
# RMSLE on train data
y_train_pred = grid_cv_rf.best_estimator_.predict(X_train)
print('RMSLE on train data: {}'.format(get_rmsle(y_train, y_train_pred)))
# RMSLE on test data
y_test_pred = grid_cv_rf.best_estimator_.predict(X_test)
print('RMSLE on test data: {}'.format(get_rmsle(y_test, y_test_pred)))
# compare several test labels and predicted labels just by view 
print(y_test[:13].to_numpy())
print(y_test_pred[:13].round())
# Display a graph of points in the space of correct labels and predictions 
# on train and test data. A diagonal cloud of point is expected for a good regression model.
# (2 scatter plots on 1 graph)
plt.figure(figsize=(7, 5))
plt.grid(True)
plt.xlim(-100,1100)
plt.ylim(-100,1100)
plt.scatter(y_train, y_train_pred, alpha=0.5, color = 'red', label='train')
plt.scatter(y_test, y_test_pred, alpha=0.5, color = 'blue', label='test')
plt.legend()
plt.title('Random Forest')
# Predictions on validation sample
y_val_pred = grid_cv_rf.best_estimator_.predict(val_data)
# Making submission file (saving results in the csv-file)
answer = pd.DataFrame({'datetime' : raw_val.datetime,
                       'count' : y_val_pred})
answer.to_csv('my_submission.csv', index=False)
print('Your submission was successfully saved!')
