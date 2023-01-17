# Loading libraries

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.cross_validation import train_test_split

#from sklearn.ensemble import AdaBoostClassifier

#from sklearn.tree import DecisionTreeClassifier
# Global constants and variables

TRAIN_FILENAME = 'train.csv'

TEST_FILENAME = 'test.csv'
train = pd.read_csv('../input/'+TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)

test = pd.read_csv('../input/'+TEST_FILENAME, parse_dates=['Dates'], index_col=False)
train.info()

test.info()
train = train.drop(['Descript', 'Resolution', 'Address'], axis = 1)
test = test.drop(['Address'], axis = 1)
def feature_engineering(data):

    data['Day'] = data['Dates'].dt.day

    data['Month'] = data['Dates'].dt.month

    data['Year'] = data['Dates'].dt.year

    data['Hour'] = data['Dates'].dt.hour

    data['Minute'] = data['Dates'].dt.minute

    data['DayOfWeek'] = data['Dates'].dt.dayofweek

    data['WeekOfYear'] = data['Dates'].dt.weekofyear

        

    data['IsWeekend'] = 0

    data.loc[data['DayOfWeek'] >4, 'IsWeekend'] = 1

    data['IsFriday'] = 0

    data.loc[data['DayOfWeek']==4, 'IsFriday'] = 1

    data['IsSaturday'] = 0

    data.loc[data['DayOfWeek']==4, 'IsSaturday'] = 1

    

    return data
train = feature_engineering(train)

test = feature_engineering(test)
from sklearn.preprocessing import LabelEncoder
dist_enc = LabelEncoder()

train['PdDistrict'] = dist_enc.fit_transform(train['PdDistrict'])
cat_enc = LabelEncoder()

cat_enc.fit(train['Category'])

train['CategoryEncoded'] = cat_enc.transform(train['Category'])

print(cat_enc.classes_)
dist_enc = LabelEncoder()

test['PdDistrict']  = dist_enc.fit_transform(test['PdDistrict'])
x_cols = list(train.columns[2:15].values)

x_cols.remove('Minute')
#from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=12)

clf = xgb.XGBClassifier(n_estimators=12, reg_alpha=0.05)
clf.fit(train[x_cols], train['CategoryEncoded'])
test['predictions'] = clf.predict(test[x_cols])
test['predictions'].unique()
# create dummy variables for each unique category

def dummy_cat(data):

    for new_col in data['Category'].unique():

        data[new_col]=(data['Category']== new_col).astype(int)

    return data
test['Category'] = cat_enc.inverse_transform(test['predictions'])

test = dummy_cat(test)
# Categories that do not get predicted need to be appended wiht the testing dataframe with

# value zero in all rows

unpredicted_cat = pd.Series(list(set(train['Category'].unique()) - (set(test['Category'].unique()))))

for new_col in unpredicted_cat:

    test[new_col] = 0
import time

PREDICTIONS_FILENAME_PREFIX = 'predictions_'

PREDICTIONS_FILENAME = PREDICTIONS_FILENAME_PREFIX + time.strftime('%Y%m%d-%H%M%S') + '.csv'
print(test.columns)
submission_cols = [test.columns[0]]+list(test.columns[17:])

print(submission_cols)
print(PREDICTIONS_FILENAME)

test[submission_cols].to_csv(PREDICTIONS_FILENAME, index = False)