# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!unzip /kaggle/input/bluebook-for-bulldozers/Train.zip

!unzip /kaggle/input/bluebook-for-bulldozers/Valid.zip
train_df = pd.read_csv('Train.csv', low_memory=False, parse_dates=["saledate"])

valid_df = pd.read_csv('Valid.csv', low_memory=False, parse_dates=["saledate"])
print(f'Shape of training dataset: {train_df.shape}')

print(f'Shape of validation dataset: {valid_df.shape}')
pd.set_option('display.max_columns', 500)
train_df.head()
X_train = train_df.drop(['SalePrice'], axis=1)

Y_train = train_df['SalePrice']
sns.set(rc={'figure.figsize':(11,8)})

sns.set(style='darkgrid')
sns.distplot(Y_train)

plt.show()
Y_train = np.log(Y_train)
pd.DataFrame((X_train.isna().sum()/len(X_train)).sort_values(ascending=False), columns=['% NaNs'])
categories = ['state', 'fiBaseModel', 'fiModelDesc', 'datasource', 'ModelID', 

              'MachineID', 'ProductGroupDesc', 'Enclosure', 'auctioneerID', 'Hydraulics', 'fiSecondaryDesc', 'Coupler',

              'Forks', 'ProductSize', 'Transmission', 'Ride_Control', 'Drive_System', 'Ripper', 'Undercarriage_Pad_Width', 

              'Thumb', 'Stick_Length', 'Pattern_Changer', 'Grouser_Type', 'Track_Type', 'Tire_Size', 'Travel_Controls', 

             'Blade_Type', 'Turbocharged', 'Stick', 'Pad_Type', 'Backhoe_Mounting', 'fiModelDescriptor', 'UsageBand', 

              'Differential_Type', 'Steering_Controls', 'fiModelSeries', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',

             'Scarifier', 'Pushblock', 'Engine_Horsepower', 'Enclosure_Type', 'Blade_Width', 'Blade_Extension', 'Tip_Control']



non_categories = ['MachineHoursCurrentMeter', 'saledate', 'YearMade']

features = categories + non_categories
X_train = X_train[features]
def to_categories(df, categories):

    for col, series in df.items():

        if col in categories: df[col] = series.astype('category').cat.as_ordered()



def fill_cat_na(df):

    for dt, col in zip(df.dtypes, df):

        if str(dt) == 'category':

            df[col] = df[col].fillna(df[col].mode().iloc[0]) 
to_categories(X_train, categories)

to_categories(X_train, [])

fill_cat_na(X_train)

X_train['MachineHoursCurrentMeter'].fillna(0, inplace=True)
any(X_train.isna().sum())
X_train.head()
def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):

    if isinstance(fldnames,str): 

        fldnames = [fldnames]

    for fldname in fldnames:

        fld = df[fldname]

        fld_dtype = fld.dtype

        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

            fld_dtype = np.datetime64



        if not np.issubdtype(fld_dtype, np.datetime64):

            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)

        targ_pre = re.sub('[Dd]ate$', '', fldname)

        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

        if time: attr = attr + ['Hour', 'Minute', 'Second']

        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

        if drop: df.drop(fldname, axis=1, inplace=True)
X_train['saledate'] = pd.to_datetime(X_train['saledate'].astype('str'))
add_datepart(X_train, 'saledate')
X_train.head()
X_train['age'] = X_train['saleYear'].astype('int64') - X_train['YearMade'].astype('int64')
Y_train.shape
params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse'},

            'subsample': 0.4,

            'subsample_freq': 1,

            'learning_rate': 0.4,

            'num_leaves': 31,

            'feature_fraction': 0.8,

            'lambda_l1': 1,

            'lambda_l2': 1

            }



folds = 10

seed = 666



kf = KFold(n_splits=folds, shuffle=False, random_state=seed)



models = []

for train_index, val_index in kf.split(X_train, Y_train):

    x_train = X_train.iloc[train_index]

    x_val = X_train.iloc[val_index]

    y_train = Y_train.iloc[train_index]

    y_val = Y_train.iloc[val_index]

    

    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categories)

    lgb_eval = lgb.Dataset(x_val, y_val, categorical_feature=categories)

    

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=(lgb_train, lgb_eval),

                early_stopping_rounds=100,

                verbose_eval = 100)

    

    models.append(gbm)
def apply_cats(df, trn):

    for n,c in df.items():

        if (n in trn.columns) and (trn[n].dtype.name=='category'):

            df[n] = c.astype('category').cat.as_ordered()

            df[n].cat.set_categories(trn[n].cat.categories, ordered=True, inplace=True)
X_test = valid_df[features]
apply_cats(X_test, X_train)

fill_cat_na(X_test)

X_test['MachineHoursCurrentMeter'].fillna(0, inplace=True)
any(X_test.isna().sum())
X_test['saledate'] = pd.to_datetime(X_test['saledate'])

add_datepart(X_test, 'saledate')
X_test['age'] = X_test['saleYear'].astype('int64') - X_test['YearMade'].astype('int64')
preds=np.exp(sum([model.predict(X_test) for model in models])/folds)
valid_actual_df = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/ValidSolution.csv', low_memory=False)
actual = valid_actual_df['SalePrice']
def rsmle(preds, actual):

    return np.sqrt(np.sum((np.log(preds) - np.log(actual))**2)/len(X_test))
score = rsmle(preds, actual)

score
for cat in categories:

    X_train[cat] = X_train[cat].cat.codes
rgsr = RandomForestRegressor(n_jobs=-1)

rgsr.fit(X_train, Y_train)

rgsr.score(X_train, Y_train)
for cat in categories:

    X_test[cat] = X_test[cat].cat.codes
preds = np.exp(rgsr.predict(X_test))

preds
score = rsmle(preds, actual)

score