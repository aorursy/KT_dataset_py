import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline


train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
print('Frequency of Categories: ', train.Category.nunique())

print('Frequency of PdDistricts: ', train.PdDistrict.nunique())

print('Frequency of DayOfWeeks: ', train.DayOfWeek.nunique())
train.info()
train = train.drop('Resolution', axis=1)

train.sample(1)
train['Date'] = pd.to_datetime(train.Dates)

test['Date'] = pd.to_datetime(test.Dates)



train = train.drop('Dates', axis=1)

test = test.drop('Dates', axis=1)

train.sample(1)
train.Date.dtype
train['IsDay'] = 0

train.loc[ (train.Date.dt.hour > 6) & (train.Date.dt.hour < 20), 'IsDay' ] = 1

test['IsDay'] = 0

test.loc[ (test.Date.dt.hour > 6) & (test.Date.dt.hour < 20), 'IsDay' ] = 1



train.sample(3)
days_to_int_dic = {

        'Monday': 1,

        'Tuesday': 2,

        'Wednesday': 3,

        'Thursday': 4,

        'Friday': 5,

        'Saturday': 6,

        'Sunday': 7,

}

train['DayOfWeek'] = train['DayOfWeek'].map(days_to_int_dic)

test['DayOfWeek'] = test['DayOfWeek'].map(days_to_int_dic)



train.DayOfWeek.unique()
train['Hour'] = train.Date.dt.hour

train['Month'] = train.Date.dt.month

train['Year'] = train.Date.dt.year

train['Year'] = train['Year']



test['Hour'] = test.Date.dt.hour

test['Month'] = test.Date.dt.month

test['Year'] = test.Date.dt.year

test['Year'] = test['Year'] 



train.sample(1)
train['HourCos'] = np.cos((train['Hour']*2*np.pi)/24 )

train['DayOfWeekCos'] = np.cos((train['DayOfWeek']*2*np.pi)/7 )

train['MonthCos'] = np.cos((train['Month']*2*np.pi)/12 )



test['HourCos'] = np.cos((test['Hour']*2*np.pi)/24 )

test['DayOfWeekCos'] = np.cos((test['DayOfWeek']*2*np.pi)/7 )

test['MonthCos'] = np.cos((test['Month']*2*np.pi)/12 )



train.sample(1)
train = pd.get_dummies(train, columns=['PdDistrict'])

test  = pd.get_dummies(test,  columns=['PdDistrict'])

train.sample(2)
from sklearn.preprocessing import LabelEncoder



cat_le = LabelEncoder()

train['CategoryInt'] = pd.Series(cat_le.fit_transform(train.Category))

train.sample(5)

cat_le.classes_
train['InIntersection'] = 1

train.loc[train.Address.str.contains('Block'), 'InIntersection'] = 0



test['InIntersection'] = 1

test.loc[test.Address.str.contains('Block'), 'InIntersection'] = 0
train.sample(10)
train.columns
feature_cols = ['X', 'Y', 'IsDay', 'DayOfWeek', 'Month', 'Hour', 'Year', 'InIntersection',

                'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE',

                'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK',

                'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN']

target_col = 'CategoryInt'



train_x = train[feature_cols]

train_y = train[target_col]



test_ids = test['Id']

test_x = test[feature_cols]
train_x.sample(1)
test_x.sample(1)
test_x.shape
train_x.shape
type(train_x), type(train_y)
import xgboost as xgb

train_xgb = xgb.DMatrix(train_x, label=train_y)

test_xgb  = xgb.DMatrix(test_x)
params = {

    'max_depth': 5,  # the maximum depth of each tree

    'eta': 0.4,  # the training step for each iteration

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'num_class': 39,

}
CROSS_VAL = False

if CROSS_VAL:

    print('Doing Cross-validation ...')

    cv = xgb.cv(params, train_xgb, nfold=3, early_stopping_rounds=10, metrics='mlogloss', verbose_eval=True)

    cv
SUBMIT = not CROSS_VAL

if SUBMIT:

    print('Fitting Model ...')

    m = xgb.train(params, train_xgb, 10)

    res = m.predict(test_xgb)

    cols = ['Id'] + cat_le.classes_

    submission = pd.DataFrame(res, columns=cat_le.classes_)

    submission.insert(0, 'Id', test_ids)

    submission.to_csv('submission.csv', index=False)

    print('Done Outputing !')

    print(submission.sample(3))

else:

    print('NOT SUBMITING')
submission.shape()