# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
train_df  = pd.read_excel("../input/train_cleaned.xlsx")
test_df = pd.read_excel("../input/test_cleaned.xlsx")
# Shape of dataset
print (train_df.shape)
print (test_df.shape)
######## Check Data Quality #############
# Check if any Stores were closed but Sales > 0 
print ('Open = 0 and Sales > 0')
print (train_df['Store'][(train_df.Sales > 0) & (train_df.Open == 0)].count())
print("----------------------------")
# Check if any Stores were open but Sales = 0
print ('Open = 1 and Sales = 0')
print (train_df['Store'][(train_df.Sales == 0) & (train_df.Open == 1)].count())
# Check n/a of train and test set
print("----------------------------")
print('train set n/a')
print(train_df.isna().sum(axis = 0))
print("----------------------------")
print('test set n/a')
print(test_df.isna().sum(axis = 0))
print("----------------------------")
#remove the records that Sales = 0
train_df = train_df[train_df['Sales'] > 0]
### Fill n/a value of "Open" --- from Store 622 to 1
test_df['Open'].fillna(1, inplace=True)
#test_features['Open'].fillna(1, inplace=True)
# Shape of dataset
print (train_df.shape)
print (test_df.shape)
######## Exploration of Data #############
fig, (axis1,axis2) = plt.subplots(1,2, figsize=(15,4))
#Average Sales by Year
sns.barplot(x='Year', y='Sales', data=train_df, ax=axis1)
# Average Sales by Month
sns.barplot(x='Month', y='Sales', data=train_df, ax=axis2)
# Average Sales by Day of Week
fig, (axis1) = plt.subplots(1, figsize=(15,4))
sns.barplot(x='DayOfWeek', y='Sales', data=train_df, ax=axis1)
# Average Sales with and without Promo 1 and Promo 2
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.countplot(x='Promo', data=train_df, ax=axis1)
sns.countplot(x='with_Promo2', data=train_df, ax=axis2)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='Promo', y='Sales', data=train_df, ax=axis1)
sns.barplot(x='with_Promo2', y='Sales', data=train_df, ax=axis2)
# Average Sales by StateHoliday and SchoolHoliday
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.countplot(x='StateHoliday', data=train_df, ax=axis1)
sns.countplot(x='SchoolHoliday', data=train_df, ax=axis2)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='StateHoliday', y='Sales', data=train_df, ax=axis1)
sns.barplot(x='SchoolHoliday', y='Sales', data=train_df, ax=axis2)
# Average Sales by StoreType and Assortment
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.countplot(x='StoreType', data=train_df, order=['a','b','c','d'],ax=axis1)
sns.countplot(x='Assortment', data=train_df, order=['a','b','c'],ax=axis2)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='StoreType', y='Sales', data=train_df, order=['a','b','c','d'],ax=axis1)
sns.barplot(x='Assortment', y='Sales', data=train_df, order=['a','b','c'],ax=axis2)
#Add store effect which equals to mean sales of the stores
mean_Sales = train_df.groupby('Store').mean()
# Scatter Plot of Average Sales vs. Effective_Competition_Distance by Store
mean_Sales.plot(kind='scatter',x='Effective_Competition_Distance',y='Sales',figsize=(15,4))
#Add Average Sales by Store to the train set, column named as 'StoreEffect'

StoreEffect = mean_Sales['Sales']
#StoreEffect = StoreEffect.rename(columns={'Sales': 'StoreEffect'})

temp_train = pd.merge(train_df, StoreEffect, how='left', on=['Store'])
temp_train
#Rename columns in train set
temp_train = temp_train.rename(columns={'Sales_x': 'Sales', 'Sales_y': 'StoreEffect'})
train_df = temp_train
train_df
#Add Average Sales by Store to the test set, column named as 'StoreEffect'
temp_test = pd.merge(test_df, StoreEffect, how='left', on=['Store'])
temp_test
#Rename columns in test set
temp_test = temp_test.rename(columns={'Sales': 'StoreEffect'})
test_df = temp_test
test_df
###Change data types to objects
train_df = train_df.astype({
    'DayOfWeek':'object',
    'SchoolHoliday':'object',
    'Month':'object'
    })

test_df = test_df.astype({
    'DayOfWeek':'object',
    'SchoolHoliday':'object',
    'Month':'object'
    })
test_df_Open = test_df[test_df['Open'] ==1]
test_df_Open.shape
test_df_Closed = test_df[test_df['Open'] ==0]
test_df_Closed.shape
train_df.info()
print("----------------------------")
test_df_Open.info()
print("----------------------------")
test_df_Closed.info()
print("----------------------------")
# Labels are the values we want to predict
#labels = np.array(train_df['Sales'])
labels = train_df.loc[:,['Sales']]
train_labels = labels
# Extract features
features = train_df.loc[:,['DayOfWeek','Promo','StateHoliday','SchoolHoliday','Month','StoreType','Assortment','with_Promo2','Effective_Competition_Distance','StoreEffect']]
train_features = features

#del features
# Using Skicit-learn to split data into training and testing sets
#from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
#train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
#print('Validation Features Shape:', valid_features.shape)
#print('Validation Labels Shape:', valid_labels.shape)
valid_labels.head(5)
#baseline_setup___simple average model
#baseline_preds = np.mean(test_labels)
#baseline_preds
#create a new df
valid_labels_base = valid_labels
valid_labels_base = valid_labels_base.assign(baseline_preds = np.mean(valid_labels_base.Sales))

valid_labels_base.head(5)
def rmse(y_true, y_pred):
    
    #Compute Root Mean Square Percentage Error between two arrays.
    
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))

    return loss
base_error = round(rmse(valid_labels_base['Sales'], valid_labels_base['baseline_preds']),4)
base_error
###one-hot encode for training set
#Object indicates a column has text (there are other things it could be theoretically be, but that's unimportant for our purposes). It's most common to one-hot encode these "object" columns, since they can't be plugged directly into most models. Pandas offers a convenient function called get_dummies to get one-hot encodings.

one_hot_encoded_training_predictors = pd.get_dummies(train_features)
train_features = one_hot_encoded_training_predictors

train_features
###one-hot encode for validation set
one_hot_encoded_valid_predictors = pd.get_dummies(valid_features)
valid_features = one_hot_encoded_valid_predictors

valid_features
# Instantiate model with 200 decision trees
rf = RandomForestRegressor(n_estimators = 200, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);
#-- Make Predictions on the Validation Set
#predictions = rf.predict(valid_features)
#-- Change the output serise to dataframe
#predictions = pd.Series(predictions)
#predictions
#--
#valid_labels_a = valid_labels.head(10000)
#valid_labels_a.head(10)
#--
#predictions_a = predictions.head(10000)
#predictions_a.head(10)
#--
#valid_labels_a.shape
#--
#predictions_a.shape
#--
#valid_labels_a = valid_labels_a[valid_labels_a.columns['Sales']]
#valid_labels_a.head(5)
#--
#rf_error = round(rmse(valid_labels_a, predictions_a),4)
#rf_error
#rf_error = round(rmse(valid_labels, predictions),4)
#rf_error
# Extract features
test_features = test_df_Open.loc[:,['DayOfWeek','Promo','StateHoliday','SchoolHoliday','Month','StoreType','Assortment','with_Promo2','Effective_Competition_Distance','StoreEffect']]
test_features
###one-hot encode for test set
one_hot_encoded_test_predictors = pd.get_dummies(test_features)
test_features = one_hot_encoded_test_predictors

test_features
### Add back the missing columns that are required by the model
test_features = test_features.assign(Month_1=0)
test_features = test_features.assign(Month_2=0)
test_features = test_features.assign(Month_3=0)
test_features = test_features.assign(Month_4=0)
test_features = test_features.assign(Month_5=0)
test_features = test_features.assign(Month_6=0)
test_features = test_features.assign(Month_7=0)
test_features = test_features.assign(Month_10=0)
test_features = test_features.assign(Month_11=0)
test_features = test_features.assign(Month_12=0)
test_features = test_features.assign(StateHoliday_b=0)
test_features = test_features.assign(StateHoliday_c=0)

test_features
# Prediction on test set
predictions_test = rf.predict(test_features)
# Assign column name as 'Sales' for predictions_test
#predictions_test = predictions_test.rename('Sales')
#type(predictions_test)
predictions_test = pd.DataFrame(data = predictions_test, columns = ['Sales'])
#Check if any null value for the prediction
predictions_test.isna().sum(axis = 0)
# Change from serise to dataframe
#predictions_test = predictions_test.to_frame()
#predictions_test
# Extract ID of Open Store of Test Set
test_features_Open_ID = test_df_Open.loc[:,['Id']]
test_features_Open_ID
#Reset index
test_features_Open_ID = test_features_Open_ID.reset_index()
#Remove redundant index column
test_features_Open_ID = test_features_Open_ID.drop(columns=['index'])
test_features_Open_ID
type(test_features_Open_ID)
#result = predictions_test
#combine prediction result with ID
#test_features_Open_ID['Sales'] = predictions_test

frames = [test_features_Open_ID, predictions_test]
submission = pd.concat(frames, axis=1)
submission
submission.shape
# Extract ID of Closed Store of Test Set
test_df_Closed_ID = test_df_Closed.loc[:,['Id']]
test_df_Closed_ID
# Add Sales = 0 for stores closed in test set
closed_submission = test_df_Closed_ID.assign(Sales=0)
closed_submission
### Append Closed_submission to submission and sort by ID
submission = submission.append(closed_submission)
submission = submission.sort_values('Id')
submission = submission.reset_index()
submission = submission.drop(columns=['index'])
submission
# save to csv file
#submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})
submission.to_csv('submission.csv', index=False)
# submission API
#kaggle competitions submit -c rossmann-store-sales -f submission.csv -m "Message"