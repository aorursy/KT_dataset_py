import pandas as pd
import re
import matplotlib as plt
import numpy as np
from datetime import *

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
%matplotlib inline
# Read the dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

dataset = train.drop(['AnimalID', 'OutcomeSubtype', 'OutcomeType'], axis=1)
dataset = dataset.append(test.drop('ID', axis=1), ignore_index=True)
print(train.shape, test.shape, dataset.shape)
dataset.isnull().sum()
#calculate Age in days
def calculate_age(x):
    if pd.isnull(x):
        return x
    num = int(x.split(' ')[0])
    if 'year' in x:
        return num * 365
    elif 'month' in x:
        return num * 30
    elif 'week' in x:
        return num * 7
    
def has_name(x):
    if pd.isnull(x):
        return 0
    return 1

def is_mix(x):
    if 'Mix' in x:
        return 1
    return 0
#data transformation

dataset['AgeuponOutcome'] = dataset['AgeuponOutcome'].apply(lambda x : calculate_age(x))
dataset['AgeuponOutcome'].fillna(dataset['AgeuponOutcome'].dropna().mean(), inplace=True)


# Since there is only one NA, I will assign it to maximum class
dataset['SexuponOutcome'].fillna('Neutered Male', inplace=True)


# Does Animal has a name
dataset['HasName'] = dataset['Name'].apply(has_name)


# Is animal of mix breed?
dataset['IsMix'] = dataset['Breed'].apply(is_mix)


# Break SexuponOutcome into two - Sterilized and Sex
sex = dataset['SexuponOutcome'].str.split(' ', expand=True)
dataset['Sterilized'] = sex[0]
dataset['Sterilized'].fillna('Unknown', inplace=True)
dataset['Sex'] = sex[1]
dataset['Sex'].fillna('Unknown', inplace=True)


dates = dataset['DateTime'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
dataset['Year'] = dates.apply(lambda x : x.year)
dataset['Month'] = dates.apply(lambda x : x.month)
dataset['Day'] = dates.apply(lambda x : x.weekday())
dataset['Hour'] = dates.apply(lambda x : x.hour)


dataset['Breed_New'] = dataset['Breed'].apply(lambda x: x.split(' Mix')[0])
breeds = dataset['Breed_New'].apply(lambda x : x.split('/'))
dataset['Breed_1'] = breeds.apply(lambda x : x[0])
# Instead of Breed_2, I will use Multiple_Breeds feature
#dataset['Breed_2'] = breeds.apply(lambda x : 'Unknown' if len(x) == 1 else x[1] )
dataset['Multiple_Breeds'] = dataset['Breed'].apply(lambda x : 1 if '/' in x else 0)


colors = dataset['Color'].apply(lambda x : x.split('/'))
dataset['Color_1'] = colors.apply(lambda x : x[0].split(' ')[0])
# Instead of Color_2, I will use Multiple_Colors feature
# dataset['Color_2'] = colors.apply(lambda x : x[1].split(' ')[0] if len(x) > 1 else 'None')
dataset['Multiple_Colors'] = dataset['Color'].apply(lambda x : 1 if '/' in x else 0)


# Encoding
enc = LabelEncoder()
dataset['Color_1'] = enc.fit_transform(dataset['Color_1'])
dataset['Breed_1'] = enc.fit_transform(dataset['Breed_1'])


# Dummy Columns
dummy_columns = ['Sterilized', 'Sex', 'AnimalType']
dataset = pd.get_dummies(dataset, columns=dummy_columns)


# Drop unnecessary columns
drop_columns = ['Name', 'DateTime', 'SexuponOutcome', 'Breed', 'Color', 'Breed_New']
dataset = dataset.drop(drop_columns, axis=1)

print(train.shape, test.shape, dataset.shape)
train_x = dataset.loc[0:26728,]

enc = LabelEncoder()
train_y = enc.fit_transform(train['OutcomeType'])
train_y = pd.DataFrame(train_y)

test_x = dataset.loc[26729:38185,]
# Cross Validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_x.values, train_y[0].values,
                                                                     test_size=0.3, random_state=0)
#Choose best parameters for randomforest
def best_params(train_x, train_y):
    rfc = RandomForestClassifier()
    param_grid = { 
        'n_estimators': [50, 400],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(train_x, train_y)
    return CV_rfc.best_params_

print(best_params(train_x.values, train_y[0].values))
# RandomForest Classifier 

rf = RandomForestClassifier(n_estimators=400, max_features='log2').fit(X_train, y_train)
print('Cross Validation for RandomForestClassifier')
print(rf.score(X_test, y_test))

prediction = pd.DataFrame(rf.predict_proba(test_x.values))
prediction.columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
prediction = pd.concat([test['ID'], prediction], axis=1)
prediction.to_csv('randomforest.csv', index=False)

#Public LeaderBoard Score - 0.81316
prediction.head()
# Simple XGBClassifier. 
# For better results we need to fine tune the parameters

xgboost = XGBClassifier(learning_rate =0.05,
 n_estimators=500,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread=4,
 scale_pos_weight=1,
 seed=27).fit(X_train, y_train)

print('Cross Validation for XGBClassifier')
print(xgboost.score(X_test, y_test))

prediction = pd.DataFrame(xgboost.predict_proba(test_x.values))
prediction.columns = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
prediction = pd.concat([test['ID'], prediction], axis=1)
prediction.to_csv('xgbclassifier.csv', index=False)

#Public LeaderBoard Score - 0.74264
prediction.head()