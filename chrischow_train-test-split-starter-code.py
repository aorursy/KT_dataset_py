# Import required modules

import numpy as np

import os

import pandas as pd

import warnings



# Settings

warnings.filterwarnings('ignore')



# Check available files

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')



# Clean text

train['Province_State'] = train['Province_State'].astype(str)

train['Province_State'] = train['Province_State'].str.replace(' ', '_').str.replace('nan', '').str.lower()

train['Country_Region'] = train['Country_Region'].astype(str).str.lower().str.replace(' ', '_')

test['Province_State'] = test['Province_State'].astype(str)

test['Province_State'] = test['Province_State'].str.replace(' ', '_').str.replace('nan', '').str.lower()

test['Country_Region'] = test['Country_Region'].astype(str).str.lower().str.replace(' ', '_')



# Use correct date format

train['Date'] = pd.to_datetime(train.Date)

test['Date'] = pd.to_datetime(test.Date)



# Create new feature for state (Country/Region + Province/State)

train['state'] = train['Country_Region'].str.lower().replace(' ', '_') + '_' + train['Province_State']

train['state'] = train['state'].str.replace('_$', '')

test['state'] = test['Country_Region'].str.lower().replace(' ', '_') + '_' + test['Province_State']

test['state'] = test['state'].str.replace('_$', '')



# Create new feature - days since 21 Jan 2020

train['day_id'] = (train.Date - pd.to_datetime('2020-01-22')).dt.days

test['day_id'] = (test.Date - pd.to_datetime('2020-01-22')).dt.days
# Train and test set IDs

TRAIN_ID1 = train.day_id[~train.day_id.isin(test.day_id)].unique()

TEST_ID1 = train.day_id[train.day_id.isin(test.day_id)].unique()



TRAIN_ID2 = train.day_id.unique()

TEST_ID2 = test.day_id[~test.day_id.isin(train.day_id)].unique()



# Targets

TGT_COLS = ['ConfirmedCases', 'Fatalities']



# Summary of parameters

print('[SUMMARY OF PARAMETERS]\n')

print('Train Set 1 IDs:    %s to %s' % ('{0:0=2d}'.format(TRAIN_ID1[0]), TRAIN_ID1[-1]))

print(' Test Set 1 IDs:    %s to %s' % (TEST_ID1[0], TEST_ID1[-1]))

print('Train Set 2 IDs:    %s to %s' % ('{0:0=2d}'.format(TRAIN_ID2[0]), TRAIN_ID2[-1]))

print(' Test Set 2 IDs:    %s to %s' % (TEST_ID2[0], TEST_ID2[-1]))
# Train and test sets

X_train1 = train[train.day_id.isin(TRAIN_ID1)].drop(TGT_COLS, axis=1)

X_train2 = train[train.day_id.isin(TRAIN_ID2)].drop(TGT_COLS, axis=1)



X_test1 = train[train.day_id.isin(TEST_ID1)].drop(TGT_COLS, axis=1)

X_test2 = test[test.day_id.isin(TEST_ID2)]



# ConfirmedCases

y1_train1 = train.ConfirmedCases[train.day_id.isin(TRAIN_ID1)]

y1_test1 = train.ConfirmedCases[train.day_id.isin(TEST_ID1)]

y1_train2 = train.ConfirmedCases[train.day_id.isin(TRAIN_ID2)]



# Fatalities

y2_train1 = train.Fatalities[train.day_id.isin(TRAIN_ID1)]

y2_test1 = train.Fatalities[train.day_id.isin(TEST_ID1)]

y2_train2 = train.Fatalities[train.day_id.isin(TRAIN_ID2)]



# Output

X_train1.to_csv('x_train1.csv', index=False)

X_train2.to_csv('x_train2.csv', index=False)

X_test1.to_csv('x_test1.csv', index=False)

X_test2.to_csv('x_test2.csv', index=False)



y1_train1.to_csv('y1_train1.csv', index=False)

y1_train2.to_csv('y1_train2.csv', index=False)

y1_test1.to_csv('y1_test1.csv', index=False)



y2_train1.to_csv('y2_train1.csv', index=False)

y2_train2.to_csv('y2_train2.csv', index=False)

y2_test1.to_csv('y2_test1.csv', index=False)