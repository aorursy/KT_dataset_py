%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Show 5 random rows from dataset

train_df.sample(5)
test_df.sample(1)
print('Number of Categories: ', train_df.Category.nunique())

print('Number of PdDistricts: ', train_df.PdDistrict.nunique())

print('Number of DayOfWeeks: ', train_df.DayOfWeek.nunique())

print('_________________________________________________')

# Show some useful Information

train_df.info()
train_df = train_df.drop('Resolution', axis=1)

train_df.sample(1)
train_df.Dates.dtype
assert train_df.Dates.isnull().any() == False

assert test_df.Dates.isnull().any() == False
assert train_df.Dates.str.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d').all() == True

assert test_df.Dates.str.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d').all() == True
train_df['Date'] = pd.to_datetime(train_df.Dates)

test_df['Date'] = pd.to_datetime(test_df.Dates)



train_df = train_df.drop('Dates', axis=1)

test_df = test_df.drop('Dates', axis=1)

train_df.sample(1)
# Confirm that it was parsed to Datetime

train_df.Date.dtype
train_df['IsDay'] = 0

train_df.loc[ (train_df.Date.dt.hour > 6) & (train_df.Date.dt.hour < 20), 'IsDay' ] = 1

test_df['IsDay'] = 0

test_df.loc[ (test_df.Date.dt.hour > 6) & (test_df.Date.dt.hour < 20), 'IsDay' ] = 1



train_df.sample(3)
days_to_int_dic = {

        'Monday': 1,

        'Tuesday': 2,

        'Wednesday': 3,

        'Thursday': 4,

        'Friday': 5,

        'Saturday': 6,

        'Sunday': 7,

}

train_df['DayOfWeek'] = train_df['DayOfWeek'].map(days_to_int_dic)

test_df ['DayOfWeek'] = test_df ['DayOfWeek'].map(days_to_int_dic)



train_df.DayOfWeek.unique()
train_df['Hour'] = train_df.Date.dt.hour

train_df['Month'] = train_df.Date.dt.month

train_df['Year'] = train_df.Date.dt.year

train_df['Year'] = train_df['Year'] - 2000 # The Algorithm doesn't know the difference. It's just easier to work like that



test_df['Hour'] = test_df.Date.dt.hour

test_df['Month'] = test_df.Date.dt.month

test_df['Year'] = test_df.Date.dt.year

test_df['Year'] = test_df['Year'] - 2000 # The Algorithm doesn't know the difference. It's just easier to work like that



train_df.sample(1)
train_df['HourCos'] = np.cos((train_df['Hour']*2*np.pi)/24 )

train_df['DayOfWeekCos'] = np.cos((train_df['DayOfWeek']*2*np.pi)/7 )

train_df['MonthCos'] = np.cos((train_df['Month']*2*np.pi)/12 )



test_df['HourCos'] = np.cos((test_df['Hour']*2*np.pi)/24 )

test_df['DayOfWeekCos'] = np.cos((test_df['DayOfWeek']*2*np.pi)/7 )

test_df['MonthCos'] = np.cos((test_df['Month']*2*np.pi)/12 )



train_df.sample(1)
train_df = pd.get_dummies(train_df, columns=['PdDistrict'])

test_df  = pd.get_dummies(test_df,  columns=['PdDistrict'])

train_df.sample(2)
from sklearn.preprocessing import LabelEncoder



cat_le = LabelEncoder()

train_df['CategoryInt'] = pd.Series(cat_le.fit_transform(train_df.Category))

train_df.sample(5)

#cat_le.classes_
train_df['InIntersection'] = 1

train_df.loc[train_df.Address.str.contains('Block'), 'InIntersection'] = 0



test_df['InIntersection'] = 1

test_df.loc[test_df.Address.str.contains('Block'), 'InIntersection'] = 0
train_df.sample(10)
train_df.columns
feature_cols = ['X', 'Y', 'IsDay', 'DayOfWeek', 'Month', 'Hour', 'Year', 'InIntersection',

                'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE',

                'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK',

                'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN']

target_col = 'CategoryInt'



train_x = train_df[feature_cols]

train_y = train_df[target_col]



test_ids = test_df['Id']

test_x = test_df[feature_cols]
train_x.sample(1)
test_x.sample(1)
type(train_x), type(train_y)
import xgboost as xgb

train_xgb = xgb.DMatrix(train_x, label=train_y)

test_xgb  = xgb.DMatrix(test_x)
params = {

    'max_depth': 4,  # the maximum depth of each tree

    'eta': 0.3,  # the training step for each iteration

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