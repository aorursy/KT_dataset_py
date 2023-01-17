# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
##Initial data understanding 
#this is training dataset. I will be creating two sample using below data sets as my training and validation dataset. This will be done after preprocessing of the data.
dataset=pd.read_csv("../input/application_train.csv")
##test dataset which is to be predicted
test=pd.read_csv("../input/application_test.csv")
#reducing the unique values in occupation by grouping by skill level. This grouping can differ based on more information about each occupation
dataset['NAME_TYPE_SUITE'].replace({'Children':'Family',
                                    'Group of people':'Other',
                                    'Other_A':'Other',
                                    'Other_B':'Other',
                                    'Spouse, partner':'Family'},inplace=True)

test['NAME_TYPE_SUITE'].replace({'Children':'Family',
                                    'Group of people':'Other',
                                    'Other_A':'Other',
                                    'Other_B':'Other',
                                    'Spouse, partner':'Family'},inplace=True)
dataset.groupby(['OCCUPATION_TYPE']).SK_ID_CURR.count()

dataset.groupby(['NAME_EDUCATION_TYPE']).SK_ID_CURR.count()
dataset.groupby(['NAME_EDUCATION_TYPE']).TARGET.mean() 
dataset['NAME_EDUCATION_TYPE'].replace({'Academic degree':'Higher education '},inplace=True)
test['NAME_EDUCATION_TYPE'].replace({'Academic degree':'Higher education '},inplace=True)
#reducing the unique values in occupation by grouping by skill level. This grouping can differ based on more information about each occupation
dataset['OCCUPATION_TYPE'].replace({'High skill tech staff':'High_Skill',
                                    'Managers':'High_Skill',
                                    'Accountants':'High_Med_Skill',
                                    'HR staff':'High_Med_Skill',
                                    'Core staff':'Med_Skill',
                                   'Cooking staff':'Med_Skill',
                                    'Realty agents':'Med_Skill',
                                    'Sales staff':'Med_Skill',
                                    'IT staff':'High_Med_Skill',
                                    'Medicine staff':'High_Med_Skill',
                                    'Secretaries':'Med_Skill',
                                    'Security staff':'Med_Skill',
                                    'Cleaning staff':'Low_Skill',
                                      'Laborers':'Low_Skill',
                                      'Low-skill Laborers':'Low_Skill',
                                      'Cleaning staff':'Low_Skill',
                                    'Waiters/barmen staff':'Low_Skill',
                                    'Private service staff':'Low_Skill',
                                    'Drivers':'Med_Skill'
                                   },inplace=True)
test['OCCUPATION_TYPE'].replace({'High skill tech staff':'High_Skill',
                                    'Managers':'High_Skill',
                                    'Accountants':'High_Med_Skill',
                                    'HR staff':'High_Med_Skill',
                                    'Core staff':'Med_Skill',
                                   'Cooking staff':'Med_Skill',
                                    'Realty agents':'Med_Skill',
                                    'Sales staff':'Med_Skill',
                                    'IT staff':'High_Med_Skill',
                                    'Medicine staff':'High_Med_Skill',
                                    'Secretaries':'Med_Skill',
                                    'Security staff':'Med_Skill',
                                    'Cleaning staff':'Low_Skill',
                                      'Laborers':'Low_Skill',
                                      'Low-skill Laborers':'Low_Skill',
                                      'Cleaning staff':'Low_Skill',
                                    'Waiters/barmen staff':'Low_Skill',
                                    'Private service staff':'Low_Skill',
                                    'Drivers':'Med_Skill'
                                   },inplace=True)
dataset.groupby(['OCCUPATION_TYPE']).SK_ID_CURR.count()
#grouping
dataset['NAME_INCOME_TYPE'].replace({'Businessman':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
test['NAME_INCOME_TYPE'].replace({'Businessman':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
print('Testing Features shape: ', test.shape)
print('Training Features shape: ', dataset.shape)
# Create a label encoder object
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
app_train=dataset
app_test=test
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in dataset:
    if dataset[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(dataset[col].unique())) <= 2:
            # Train on the training data
            le.fit(dataset[col])
            # Transform both training and testing data
            app_train[col] = le.transform(dataset[col])
            app_test[col] = le.transform(test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
app_train= pd.get_dummies(app_train)
app_test= pd.get_dummies(app_test)
app_train.dtypes.value_counts()
app_test.dtypes.value_counts()
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_train.fillna(dataset.median(),inplace = True)

app_test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
app_test.fillna(dataset.median(),inplace = True)
#replace all  NaN in the var_list with zero
Var_List=('OBS_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
        'DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
         'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR')
def missing_val_replace(data,Var_List):
    for col in data:
        for i in Var_List:
            if col==i:
                data[col].fillna(0)
                print (col)
    return data
app_train=missing_val_replace(app_train,Var_List) 
#replace all other NaN with median values
app_train=app_train.fillna(app_train.median)
app_test=missing_val_replace(app_test,Var_List) 
#replace all other NaN with median values
app_test=app_test.fillna(app_test.median)

app_train.dtypes.value_counts()
app_test.dtypes.value_counts()

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
app_trainv2=app_train
# Add the target back in
#app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Training Features shape: ', app_trainv2.shape)
print('Testing Features shape: ', app_test.shape)
from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns=['TARGET'])
else:
    train = app_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, train_labels)
# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]
# Submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
# Save the submission to a csv file
submit.to_csv('log_reg_baseline.csv', index = False)
from sklearn.ensemble import RandomForestClassifier

# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
# Train on the training data
random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = random_forest.predict_proba(test)[:, 1]
 #Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions
submit.to_csv('random_forest_baseline.csv', index = False)
