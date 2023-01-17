# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv ('/kaggle/input/titanic/train.csv')
test = pd.read_csv ('/kaggle/input/titanic/test.csv')

print ('Train Dataset')
for feature in train.columns:
    print (feature, 'has ', train[feature].isnull().sum(), ' no of missing values')

print ('\n\n Test Dataset')
for feature in test.columns:
    print (feature, 'has ', test[feature].isnull().sum(), ' no of missing values')
def replace_missing_ages_train (data):
    data_copy = data.copy ()
    data_copy ['Surv'] = np.where (data['Survived'] == 0, 'Not Survived', 'Survived')
    sns.boxplot (x = 'Pclass', y = 'Age', hue = 'Surv', data = data_copy)

    group = data_copy.groupby(['Pclass','Survived'])['Age'].median()
    age_isnull = data['Age'].isnull()
    for index in age_isnull.index:
        if age_isnull[index] == True:
            pclass = data['Pclass'].iloc[index]
            survived = data['Survived'].iloc[index]
            data ['Age'].iloc[index] = group[pclass][survived]
    return data

def replace_missing_ages_test (data):
    data_copy = data.copy ()
    group = data_copy.groupby('Pclass')['Age'].median()
    age_isnull = data['Age'].isnull()
    for index in age_isnull.index:
        if age_isnull[index] == True:
            pclass = data['Pclass'].iloc[index]
            data ['Age'].iloc[index] = group[pclass]
    return data
        
train = replace_missing_ages_train (train)
test = replace_missing_ages_test (test)
print (test['Age'].isnull().sum())
train.drop (columns = ['Cabin'], inplace = True)
test.drop (columns = ['Cabin'], inplace = True)

train ['Embarked'][train['Embarked'].isnull()] = 'Q'
test ['Fare'][test['Fare'].isnull()] = test['Fare'].median()

print ('Train Dataset')
for feature in train.columns:
    print (feature, 'has ', train[feature].isnull().sum(), ' no of missing values')

print ('\n\n Test Dataset')
for feature in test.columns:
    print (feature, 'has ', test[feature].isnull().sum(), ' no of missing values')
def handle_outliers (data, indexes):
    perc_75 = np.percentile (data, 75)
    perc_25 = np.percentile (data, 25)
    IQR = perc_75 - perc_25
    
    limit_up = perc_75 + 1.5*IQR
    limit_low = perc_25 - 1.5*IQR
    
    #outliers = []
    for index in indexes:
        if data.iloc[index] > limit_up:
            #outliers.append (data.iloc[index])
            data.iloc[index] = limit_up
        if data.iloc[index] < limit_low:
            #outliers.append (data.iloc[index])
            data.iloc[index] = limit_low
    #print (len(outliers))
    
handle_outliers (train['Age'], train.index)
handle_outliers (train['Fare'], train.index)
handle_outliers (test['Age'], test.index)
handle_outliers (test['Fare'], test.index)
train.drop (columns = ['PassengerId', 'Name', 'Ticket'], inplace = True)
test.drop (columns = ['PassengerId', 'Name', 'Ticket'], inplace = True)

train = pd.get_dummies (train, columns = ['Sex', 'Embarked'], drop_first = True)
test = pd.get_dummies (test, columns = ['Sex', 'Embarked'], drop_first = True)

sel_features = test.columns
def perform_min_max_scaling (data):
    scaler = MinMaxScaler ()
    dataset = data [sel_features]
    scaler.fit (dataset)
    #print (dataset.head(5))
    transform = scaler.transform (dataset)
    temp_df = pd.DataFrame (transform, columns = sel_features)
    #print (temp_df.head (5))
    return temp_df
    
train_scaled = pd.concat ([train['Survived'], perform_min_max_scaling (train)], axis = 1)
test_scaled = perform_min_max_scaling (test)

print (train_scaled.head (5))
print (test_scaled.head (5))

train_scaled.to_csv ('/kaggle/working/train_ready.csv')
test_scaled.to_csv ('/kaggle/working/test_ready.csv')