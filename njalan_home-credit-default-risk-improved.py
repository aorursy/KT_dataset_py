# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the main data file "application_train/test.csv" for baseline model

train_data = pd.read_csv("../input/application_train.csv")

print('training data shape: ', train_data.shape)



test_data = pd.read_csv("../input/application_test.csv")

print('testing data shape: ', test_data.shape)
# Looking at different features in training data

train_data.head()
# Looking at overall statistics of training data

train_data.describe()
#Lets first drop the target column from the train data

target = train_data['TARGET']

#train_data = train_data.drop(columns = ['TARGET'])

#print('training data shape: ', train_data.shape)
# Lets look the the target distribution in the data set

target.value_counts()
# Now plot the histogram to visualize it further

target.plot.hist()
train_data.dtypes.value_counts()
# Utility function to change Object type training data to categorical data

def one_hot_encoding(train_data, test_data):

    """

    examine columns with object type in training and test data

    do one hot encoding of such columns

    """



    encoded_train_data = pd.get_dummies(train_data)

    encoded_test_data = pd.get_dummies(test_data)

    return encoded_train_data, encoded_test_data
encoded_train_data, encoded_test_data = one_hot_encoding(train_data, test_data)

print('Training Features shape after one hot encoding: ', encoded_train_data.shape)

print('Testing Features shape after one hot encoding: ', encoded_test_data.shape) 
align_train_data, align_test_data = encoded_train_data.align(encoded_test_data, join = 'inner', axis = 1)

align_train_data['TARGET'] = target

print('Training Features shape: ', align_train_data.shape)

print('Testing Features shape: ', align_test_data.shape)
missing_val_count_by_column = align_train_data.isnull().sum()

missing_val_count_by_column = missing_val_count_by_column[missing_val_count_by_column > 0]

print('Number of columns with missing values: ', missing_val_count_by_column.shape[0])

missing_val_count_by_column.head()
align_train_data.describe()
align_test_data.describe()
col_with_mean = align_train_data.mean(axis = 0)

col_with_negative_mean = col_with_mean[col_with_mean < 0]

col_with_negative_mean
col_name_with_negative_mean = ['DAYS_BIRTH','DAYS_REGISTRATION','DAYS_ID_PUBLISH','DAYS_LAST_PHONE_CHANGE']

align_train_data[col_name_with_negative_mean] = align_train_data[col_name_with_negative_mean]/-365

align_test_data[col_name_with_negative_mean] = align_test_data[col_name_with_negative_mean]/-365

align_train_data.describe()
import matplotlib.pyplot as plt

align_train_data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
col_with_anom = align_train_data['DAYS_EMPLOYED'] >= 36500

print('Number of anomalies : ', col_with_anom.sum())

align_train_data['DAYS_EMPLOYED'][col_with_anom].value_counts()
# Replace the anomalous values with nan

align_train_data['DAYS_EMPLOYED'].replace({365243 : np.nan}, inplace=True)

align_test_data['DAYS_EMPLOYED'].replace({365243 : np.nan}, inplace=True)

align_train_data['DAYS_EMPLOYED'].describe()
align_train_data['DAYS_EMPLOYED'] = abs(align_train_data['DAYS_EMPLOYED'])

align_test_data['DAYS_EMPLOYED'] = abs(align_test_data['DAYS_EMPLOYED'])



#Now plot histogram to visualize days of employment distribution

align_train_data['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');

plt.xlabel('Days Employment');
correlations = align_train_data.corr()['TARGET'].sort_values()

# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
feature_col = ['EXT_SOURCE_3','EXT_SOURCE_2','EXT_SOURCE_1','DAYS_BIRTH','DAYS_EMPLOYED']

X = align_train_data[feature_col]

Y = align_train_data['TARGET']

X_test = align_test_data[feature_col]

print('Training Features shape : ', X.shape)

print('Testing Features shape : ', X_test.shape) 
# have a final look at train  data to check if we miss something

X.describe()
# have a final look at test data to check if we miss something

X_test.describe()
# Utility functions to fix missing values    

def impute_missing_values(train_data, test_data):

    """

    check for the missing values and impute it with mean values of the columns

    in both train and test data

    """

    from sklearn.preprocessing import Imputer

    imputer = Imputer(strategy = 'median')

    imputed_train_data = imputer.fit_transform(train_data)

    imputed_test_data = imputer.transform(test_data)

    return imputed_train_data, imputed_test_data

#impute the missing values

X, X_test = impute_missing_values(X, X_test)

print('training data shape after imputing missing values: ', X.shape)

print('testing data shape after imputing missing values: ', X_test.shape)
# Utility functions to normalize data   

def normalize_values(train_data, test_data):

    """

    Normalizing the feature data values using MinMaxScaler library 

    in both train and test data

    """

    

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range = (0, 1)) 

    normalized_train_data = scaler.fit_transform(train_data)

    normalized_test_data = scaler.transform(test_data)

    return normalized_train_data, normalized_test_data
#normalize data

X, X_test = normalize_values(X, X_test)

print('training data shape after normalizing values: ', X.shape)

print('testing data shape after normalizing values: ', X_test.shape)
#Utility function to create logistic regression model

import time   

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

    

def create_random_forest_model(X, Y):

    """

    split data into training and validation data, for both features and target

    The split is based on a random number generator. Supplying a numeric value to

    the random_state argument guarantees we get the same split every time we

    run this script.

    

    create model using sklearn random forest library and measuere model performance

    """



    print('Starting random forest model training...')

    t0 = time.time()

    

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state = 0, test_size=0.2)



    # Make the random forest classifier

    model = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)



    # Train on the training data

    model.fit(X_train, Y_train)

    

    accuracy = model.score(X_val, Y_val)

    print('Accuray of random forest model is : ', accuracy)

    

    Y_pred = model.predict(X_val)

    mae = mean_absolute_error(Y_val, Y_pred)

    print('mean absoute error of random forest model is : ', mae)

    

    t1 = time.time()

    print('Time elapsed during random forest model training is : ', t1-t0)

    

    return model
# create logistic regression model

random_forest_model = create_random_forest_model(X,Y)
from xgboost import XGBRegressor



def create_xgboost_model(X, Y):

    """

    split data into training and validation data, for both features and target

    The split is based on a random number generator. Supplying a numeric value to

    the random_state argument guarantees we get the same split every time we

    run this script.

    

    create model using xgboost library and measuere model performance

    """



    print('Starting xgboost model training...')

    t0 = time.time()

    

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state = 0, test_size=0.2)



    # Make the random forest classifier

    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)



    # Train on the training data

    model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=[(X_val, Y_val)], verbose=False)

    

    accuracy = model.score(X_val, Y_val)

    print('Accuray of xgboost model is : ', accuracy)

    

    Y_pred = model.predict(X_val)

    mae = mean_absolute_error(Y_val, Y_pred)

    print('mean absoute error of xgboost model is : ', mae)

    

    t1 = time.time()

    print('Time elapsed during xgboost model training is : ', t1-t0)

    

    return model
#xgboost_model = create_xgboost_model(X,Y)
# generate the prediction for test data

# Make sure to select the second column only

random_forest_pred = random_forest_model.predict_proba(X_test)[:,1]
# Submission dataframe

submit = align_test_data[['SK_ID_CURR']]

submit['TARGET'] = random_forest_pred

submit.head()
# Save the submission to a csv file

submit.to_csv('random_forest_baseline.csv', index = False)