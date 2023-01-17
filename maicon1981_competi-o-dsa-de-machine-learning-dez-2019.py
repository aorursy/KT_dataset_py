# Import libraries to store data

import pandas as pd

import numpy as np



# Import libraries to visualize data

import matplotlib.pyplot as plt

import seaborn as sns



# Feature selection

from boruta import BorutaPy



# Import libraries to process data

from sklearn.preprocessing import StandardScaler



# Import libraries to classify data and score results

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# Import libraries used in functions and for feedback

import os

import gc

import logging

import warnings

import pandas_profiling
# Settings

path = os.getcwd()

gc.enable()

%matplotlib inline

pd.options.display.max_seq_items = 150

pd.options.display.max_rows = 150

pd.set_option('display.max_columns', None)

warnings.filterwarnings("ignore")
# Kaggle kernel: IS_LOCAL = False

IS_LOCAL = False

if(IS_LOCAL):

    PATH='../input/'

else:

    PATH='../input/competicao-dsa-machine-learning-dec-2019/'
print(os.listdir(PATH))
# Logger

def get_logger():

    FORMAT = '[%(levelname)s] %(asctime)s: %(name)s: %(message)s'

    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger('Main')

    logger.setLevel(logging.DEBUG)

    return logger



logger = get_logger()
logger.info('Start load data')
# Read in data into a dataframe

train = pd.read_csv(os.path.join(PATH, 'dataset_treino.csv'))

test = pd.read_csv(os.path.join(PATH, 'dataset_teste.csv'))
logger.info('Start exploratory data analysis')
# Show dataframe columns

print(train.columns)
# Display top of dataframe

train.head()
# Display the shape of dataframe

train.shape
# See the column data types and non-missing values

train.info()
# See the column data types

train.dtypes
# Unique values by features

train.nunique(dropna=False, axis=0)
# Missing values by features

train.isnull().sum(axis=0)
# Statistics of numerical features

train.describe().T
# Correlation map for train dataset

corr = train.corr()

_ , ax = plt.subplots( figsize =( 18 , 18 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = False, annot_kws = {'fontsize' : 12 })
'''

# Using pandas_profiling

# Display the report in a Jupyter notebook

profile_train = train.profile_report(style={'full_width':True}, title='Pandas Profiling Report')

profile_train

'''
'''

# Generate a HTML report file

#profile_train.to_file(output_file="train.html")

'''
# Show dataframe columns

print(test.columns)
# Display top of dataframe

test.head()
# Display the shape of dataframe

test.shape
# See the column data types and non-missing values

test.info()
# See the column data types

test.dtypes
# Unique values by features

test.nunique(dropna=False, axis=0)
# Missing values by features

test.isnull().sum(axis=0)
# Statistics of numerical features

test.describe().T
# Correlation map for train dataset

corr = test.corr()

_ , ax = plt.subplots( figsize =( 18 , 18 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = False, annot_kws = {'fontsize' : 12 })
'''

# Using pandas_profiling

# Display the report in a Jupyter notebook

profile_test = test.profile_report(style={'full_width':True}, title='Pandas Profiling Report')

profile_test

'''
'''

# Generate a HTML report file

#profile_test.to_file(output_file="test.html")

'''
logger.info('Start feature engineering')
# Preprocessing before data engineering

train_dim = train.shape[0]



# Predict feature

predict = train['target']



# New dataframe

data = pd.concat([train, test]).reset_index(drop=True)



# Drop the columns

data = data.drop(['ID', 'target'], axis=1)
# See the column data types and non-missing values

data.info()
logger.info('Start processing missing values')
# Function to calculate missing values by column

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(data)
 # Fill missing data with 'None'

for col in data.columns[data.dtypes == 'object']:

    data[col] = data[col].fillna('None')
# Fill missing data with media

data.fillna(data.mean(), inplace=True)
missing_values_table(data)
# Label encoding

for col in data.columns[data.dtypes == 'object']:

    data[col] = data[col].factorize()[0]
# Kaggle: divide dataset of train and test

train = data[:train_dim]

test = data[train_dim:]



# Append the predict feature

train['target'] = predict
# See the column data types and non-missing values

train.info()
# See the column data types and non-missing values

test.info()
# Separating predictors and target features

X = train.drop(['target'], axis=1)

y = train['target']



X = X.values

y = y.values

y = y.ravel()



# Random Forest Classifier

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

rf.fit(X, y)



# Define the feature selection method

feat_selector = BorutaPy(rf, n_estimators=100, verbose=2, random_state=1)



# Search for all relevant features

feat_selector.fit(X, y)



# Check selected features

feat_selector.support_



# Check feature ranking

feat_selector.ranking_



# Call call transform () on training data to filter features

X_train_filtered = feat_selector.transform(X)



# Show the dataset shape

X_train_filtered.shape
# Test dataset

X_test = test

X_test = X_test.values

X_test_filtered = feat_selector.transform(X_test)

X_test_filtered.shape
logger.info('Scaling features')
# Separate out the features and targets

X_train = X_train_filtered.copy()

y_train = train['target']

X_test = X_test_filtered.copy()
'''

# Separate out the features and targets

X_train = train.drop(['target'], axis=1)

y_train = train['target']

X_test = test

'''
X_train.shape
X_test.shape
y_train.shape
# Create the scaler object

scaler = StandardScaler()

# Fit on the training data

scaler.fit(X_train)

# Transform both the training and testing data

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
gc.collect()
logger.info('Prepare model')
!nvidia-smi
# XGBoost

def run_model(model, X_tr, y_tr, useTrainCV=True, cv_folds=5, early_stopping_rounds=10):

    

    # Cross-Validation

    if useTrainCV:

        xgb_param = model.get_xgb_params()

        xgtrain = xgb.DMatrix(X_tr, label=y_tr)

        

        print ('Start cross validation')

        cvresult = xgb.cv(xgb_param, 

                          xgtrain, 

                          num_boost_round=model.get_params()['n_estimators'], 

                          nfold=cv_folds,

                          metrics=['logloss'],

                          stratified=True,

                          verbose_eval=True,

                          early_stopping_rounds=early_stopping_rounds)



        model.set_params(n_estimators=cvresult.shape[0])

        best_tree = cvresult.shape[0]

        print('Best number of trees = {}'.format(best_tree))

        

    # Fit

    model.fit(X_tr, y_tr, eval_metric='logloss')

        

    # Prediction

    train_pred = model.predict(X_tr)

    train_pred_prob = model.predict_proba(X_tr)[:,1]

    

    # Log and Chart

    print("Log Loss (Train): %f" % log_loss(y_tr, train_pred_prob))

    print("Log Loss (Test): %f" % cvresult['test-logloss-mean'][best_tree-1])

    

    feature_imp = pd.Series(model.feature_importances_.astype(float)).sort_values(ascending=False)

    

    plt.figure(figsize=(18,8))

    feature_imp[:25].plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')

    plt.tight_layout()
# Model params

modelXGB = XGBClassifier(learning_rate = 0.1,

                         n_estimators = 300,

                         max_depth = 5,

                         min_child_weight = 1,

                         gamma = 0,

                         subsample = 0.8,

                         colsample_bytree = 0.8,

                         objective = 'binary:logistic',

                         n_jobs = -1,

                         tree_method = 'gpu_hist',

                         scale_pos_weight = 1)
print(modelXGB)
# Run model

run_model(modelXGB, X_train, y_train)
# Make predictions in testing dataset

test_pred_prob = modelXGB.predict_proba(X_test)[:,1]
gc.collect()
# Defining the parameters that will be tested in GridSearch

# max_depth e min_child_weight

param_v1 = {

    'max_depth':range(2,10),

    'min_child_weight':range(1,6)

}



grid_1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 300, 

                                                max_depth = 5,

                                                min_child_weight = 1, 

                                                gamma = 0, 

                                                subsample = 0.8, 

                                                colsample_bytree = 0.8,

                                                objective = 'binary:logistic',

                                                tree_method = 'gpu_hist',

                                                scale_pos_weight = 1),

                                                param_grid = param_v1, 

                                                scoring = 'neg_log_loss',

                                                n_jobs = -1,

                                                verbose = 1,

                                                iid = False, 

                                                cv = 5)



# Fit and get the best grid parameters

grid_1.fit(X_train, y_train)

grid_1.best_params_, grid_1.best_score_
gc.collect()
# Defining the parameters that will be tested in GridSearch

# gamma

param_v2 = {

    'gamma':[i/10.0 for i in range(0,10)]

}



grid_2 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 300, 

                                                max_depth = grid_1.best_params_['max_depth'],

                                                min_child_weight = grid_1.best_params_['min_child_weight'], 

                                                gamma = 0, 

                                                subsample = 0.8, 

                                                colsample_bytree = 0.8,

                                                objective = 'binary:logistic',

                                                tree_method = 'gpu_hist',

                                                scale_pos_weight = 1),

                                                param_grid = param_v2, 

                                                scoring = 'neg_log_loss',

                                                n_jobs = -1,

                                                verbose = 1,

                                                iid = False, 

                                                cv = 5)



# Fit and get the best grid parameters

grid_2.fit(X_train, y_train)

grid_2.best_params_, grid_2.best_score_
gc.collect()
# Defining the parameters that will be tested in GridSearch

# subsample e colsample_bytree

param_v3 = {

    'subsample':[i/10.0 for i in range(4,10)],

    'colsample_bytree':[i/10.0 for i in range(4,10)]

}



grid_3 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 300, 

                                                max_depth = grid_1.best_params_['max_depth'],

                                                min_child_weight = grid_1.best_params_['min_child_weight'], 

                                                gamma = grid_2.best_params_['gamma'], 

                                                subsample = 0.8, 

                                                colsample_bytree = 0.8,

                                                objective = 'binary:logistic',

                                                tree_method = 'gpu_hist',

                                                scale_pos_weight = 1),

                                                param_grid = param_v3, 

                                                scoring = 'neg_log_loss',

                                                n_jobs = -1,

                                                verbose = 1,

                                                iid = False, 

                                                cv = 5)



# Fit and get the best grid parameters

grid_3.fit(X_train, y_train)

grid_3.best_params_, grid_3.best_score_
gc.collect()
# Defining the parameters that will be tested in GridSearch

# reg_alpha

param_v4 = {

    'reg_alpha':[0, 0.001, 0.005]

}



grid_4 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 

                                                n_estimators = 300, 

                                                max_depth = grid_1.best_params_['max_depth'],

                                                min_child_weight = grid_1.best_params_['min_child_weight'], 

                                                gamma = grid_2.best_params_['gamma'], 

                                                subsample = grid_3.best_params_['subsample'], 

                                                colsample_bytree = grid_3.best_params_['colsample_bytree'],

                                                objective = 'binary:logistic',

                                                tree_method = 'gpu_hist',

                                                scale_pos_weight = 1),

                                                param_grid = param_v4, 

                                                scoring = 'neg_log_loss',

                                                n_jobs = -1,

                                                verbose = 1,

                                                iid = False, 

                                                cv = 5)



# Fit and get the best grid parameters

grid_4.fit(X_train, y_train)

grid_4.best_params_, grid_4.best_score_
gc.collect()
# Creating the XGB model with all optimizations

# reducing Learning Rate and increasing the number of estimators

modelXGB_v2 = XGBClassifier(learning_rate = 0.001, 

                            n_estimators = 40000, 

                            max_depth = grid_1.best_params_['max_depth'],

                            min_child_weight = grid_1.best_params_['min_child_weight'],

                            gamma = grid_2.best_params_['gamma'], 

                            subsample = grid_3.best_params_['subsample'],

                            colsample_bytree = grid_3.best_params_['colsample_bytree'],

                            reg_alpha = grid_4.best_params_['reg_alpha'], 

                            objective = 'binary:logistic', 

                            n_jobs = -1,

                            tree_method = 'gpu_hist',

                            scale_pos_weight = 1)
print(modelXGB_v2)
# Run model

run_model(modelXGB_v2, X_train, y_train)
# Make predictions in testing dataset

test_pred_prob = modelXGB_v2.predict_proba(X_test)[:,1]
logger.info("Prepare submission")
# Read in data into a dataframe

submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
submission['PredictedProb'] = test_pred_prob.reshape(test_pred_prob.shape[0])
submission.to_csv('submission.csv', index=False)
submission.head(20)