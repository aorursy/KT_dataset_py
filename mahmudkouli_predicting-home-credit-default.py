# Import libraries
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,GradientBoostingClassifier, AdaBoostClassifier
print(os.listdir("../input"))

application_train = pd.read_csv('../input/application_train.csv')
application_test = pd.read_csv('../input/application_train.csv')
description_train = pd.DataFrame(application_train.describe())
print(description_train)
print('Shape of the dataset is', application_train.shape)
print(application_train.columns)

def convert_days_into_years(df, columns):
    '''Converts days since an event happened to years in absolute value'''
    for column in columns:
        df[column] = np.floor((df[column] / 365).abs())

def replace_nulls_string(df, columns_):
    '''Converts NaNs in columns with type string into "Not Specified" '''
    for column in columns_:
        if df[column].dtype == 'object':
            try:
                df[column].replace(np.nan, 'Not specified', inplace=True)
            except:
                print('Check back, there is an error in string')
                
def replace_nulls_float(df, columns_):
    '''Converts NaNs in columns with type float into 0'''
    for column in columns_:
        if df[column].dtype == 'float64':
            try:
                df[column].replace(np.nan, 0, inplace=True)
                df[column].replace(1000, 0, inplace=True)
            except:
                print('Check back, there is an error in float')
    
def fillnan_mean(df, columns_):
    '''Replaces NaN values by the mean of the feature'''
    for column in columns_:
        if df[column].count() != len(df[df.columns.max()]):
            df[column].fillna(df[column].mean(), inplace=True)
            print(column, 'sucessfully changed')
            
def check_obs_count(df, columns_):
    '''Checks given columns for missing values and returns a DataFrame with all columns with missing values.'''
    column_name = ['Number of Values']
    missing_cols = []
    missing_cols_rows = []
    for column in columns_:
        if df[column].count() != len(df[df.columns.max()]):
            missing_cols.append(df[column].name)
            missing_cols_rows.append(df[column].count())
    dataframe = pd.DataFrame({'Feature': missing_cols, 'Number of Rows': missing_cols_rows})
    
    return dataframe

def get_feature_types(df, features):
    '''Checks the data type in the feature column list and returns two list, with numerical and object features'''
    numerical_features = []
    text_features = []
    for feature in features:
        if (df[feature].dtype == 'float64' or df[feature].dtype == 'int64'):
            numerical_features.append(df[feature].name)
        
        elif df[feature].dtype == 'object':
            text_features.append(df[feature].name)
        
        else:
            print('Unknown type', df[feature].dtype, 'in feature', feature)
        
    return numerical_features, text_features
feature_list = ['NAME_CONTRACT_TYPE', 'CODE_GENDER','FLAG_OWN_CAR','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT',
                 'AMT_ANNUITY','AMT_GOODS_PRICE','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                 'NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','OWN_CAR_AGE','FLAG_OWN_REALTY', 
                 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL',
                'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 
                'WEEKDAY_APPR_PROCESS_START', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_13',
                'AMT_REQ_CREDIT_BUREAU_YEAR']

convert_days_into_years(application_train, ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'])

fillnan_mean(application_train, ['AMT_ANNUITY', 'AMT_GOODS_PRICE'])

replace_nulls_float(application_train, ['DAYS_EMPLOYED', 'OWN_CAR_AGE', 'AMT_REQ_CREDIT_BUREAU_YEAR'])

replace_nulls_string(application_train, ['NAME_TYPE_SUITE', 'OCCUPATION_TYPE'])

count = check_obs_count(application_train, feature_list)

numerical_features, text_features = get_feature_types(application_train, feature_list)
# select features
X_train = application_train[feature]

# select target
y_train = application['TARGET']

# obtain dummy variables for text features
X_train = pd.get_dummies(data=X_train, columns=text_features)

# start thinking about which models to use (including tuning the hyperparameters within the final implementation of all models)
# write a function that records the accuracy of each model and then compare them