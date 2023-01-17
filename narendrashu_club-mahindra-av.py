# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#!pip install fastai==0.7

import fastai



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

import os

print(os.listdir("../input"))

from fastai.imports import *

# from fastai.structured import *

# from fastai.column_data import *



from fastai.tabular import *

# Any results you write to the current directory are saved as output.

fastai.__version__
train = pd.read_csv('../input/train-mahindra/train.csv')

test = pd.read_csv('../input/test_mahindra/test.csv')

submit = pd.read_csv('../input/sample_submission_dlc0jkw/sample_submission.csv')
## To display all the data 

## used from fastai

def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
## add columns relevant in a date column

## used from fastai

def add_datepart(df, fldname, drop=True, time=False):

    "Helper function that adds columns relevant to a date."

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)
## to download csv file from the kernel

from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)
## helper function to fix missing values

## creates an extra column indicating if that row is missing from the data or not

## taken from  fastai

def fix_missing(df, col, name, na_dict):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict
train.shape, test.shape, train.shape[0] + test.shape[0], submit.shape
display_all(train.head())
test.head()
submit.head()
## lets check the datatype of the columns

train.info()
## check for missing values in a column

train.isnull().sum(), test.isnull().sum()
## let's introduce target column in test as well for easier feature engineering

#y = train.amount_spent_per_room_night_scaled

#X = train.drop(['amount_spent_per_room_night_scaled'], axis = 1)

test['amount_spent_per_room_night_scaled'] = 0.00
## give label to data based on train/test and concat both data set

train['type'] = 'train'

test['type'] = 'test'

data = pd.concat([train, test], axis = 0, ignore_index= True)  
## let's fix the missing values first we noticed in 2 columns

fix_missing(data, data['season_holidayed_code'], 'season_holidayed_code', {})

fix_missing(data, data['state_code_residence'], 'state_code_residence', {})

data['season_holidayed_code'] = data['season_holidayed_code'].astype('int')

data['state_code_residence'] = data['state_code_residence'].astype('int')
## creating few columns like total people stayed (adults + children), total adult night, total people night

data['total_people'] = data['numberofadults'] + data['numberofchildren']

data['total_adult_night'] = data['numberofadults'] * data['roomnights']

data['total_people_night'] = data['total_people'] * data['roomnights']
## convert data columns to date format

# data['booking_date'] = pd.to_datetime(data['booking_date'], format= '%d/%m/%y')

# data['checkin_date'] = pd.to_datetime(data['checkin_date'], format= '%d/%m/%y')

# data['checkout_date'] = pd.to_datetime(data['checkout_date'], format= '%d/%m/%y')



## create advance booking days = checkin date - booking date

data['days_advance']=(pd.to_datetime(data['checkin_date'],format= '%d/%m/%y')-pd.to_datetime(data['booking_date'],format= '%d/%m/%y')).dt.days



## actual stays based on checkin and checkout = checkout date - checkin date

data['actual_stay']=(pd.to_datetime(data['checkout_date'],format= '%d/%m/%y')-pd.to_datetime(data['checkin_date'],format= '%d/%m/%y')).dt.days
## lets see the statistics

data.describe()
## room night has a minimum value which is negative. lets see all the rows

data[data['roomnights'] < 0]
## its just a row.  replace the values manually based on previus calculation  

data['roomnights'] = data['roomnights'].replace(-45, 7)

data['total_adult_night'] = data['total_adult_night'].replace(-180, 28)

data['total_people_night'] = data['total_people_night'].replace(-315, 49)
## we also see that days advance is negative for some roows. lets see them

data[data['days_advance'] < 0]
## it seems like there may be an error in these rows as booking date in in 2018 while stays in 2012. 

## we have 14 such rows. lets replace those by 0 anyway as those will not impact very much

#data[data['days_advance']<0]['days_advance'] = 0

# data['days_advance'][data['days_advance'] < 0] = 0

data.days_advance=data.days_advance.mask(data.days_advance.lt(0),0)
## create a column overstay to represent if the actual stay (based on checkin and check out date) is greater or less or equal to 

## given in the data

conditions = [

    (data['actual_stay'] > data['roomnights']),

    (data['actual_stay'] < data['roomnights']),

    (data['actual_stay'] == data['roomnights'])]

choices = ['overstay', 'understay', 'rightstay']

data['over_stay'] = np.select(conditions, choices)
## how many of them stays more days than  booked days

data['stay_over'] = data['actual_stay'] - data['roomnights']
## this 'stay_over' range from -78 days to 22 days. lets create these into groups

bins = [-100, -50, -25, -10, -0.1, 0.1, 5, 10, 15, 20, 30]

names = ['group1', 'group2', 'group3', 'group4', 'group5', 'group6','group7', 'group8', 'group9', 'group10']

data['stay_over_group'] = pd.cut(data['stay_over'], bins, labels=names)
## we see that advance booking has been done prior to 177 days max. lets group these days to form a different column.

## categorising them in terms of group for eg. groupA has days_advance of 0 days, B has days advance booking from 0 to 7 days

bins = [-0.01, 0.01, 7, 15, 30, 60, 90, 120, 180, 210]

names = ['groupA', 'groupB', 'groupC', 'groupD', 'groupE', 'groupF','groupG', 'groupH', 'groupI']

data['advance_booking_group'] = pd.cut(data['days_advance'], bins, labels=names)
## total person travelling differences: total adults+total children - total pax

data['total_pax_differ'] = data['total_people'] - data['total_pax']

## it ranges from -22 days to 32 days
## lets group them as well

bins = [-25, -20, -15, -10, -5, -0.1, 0.1, 5, 10, 15, 20,25,30, 35]

names = ['groupa', 'groupb', 'groupc', 'groupd', 'groupe', 'groupf','groupg', 'grouph', 'groupi', 'groupj', 'groupk','groupl', 'groupm']

data['total_pax_differ_group'] = pd.cut(data['total_pax_differ'], bins, labels=names)
## function to create unicodes based on combinations of different columns

def make_identifier(df):

    str_id = df.apply(lambda x: '_'.join(map(str, x)), axis=1)

    return pd.factorize(str_id)[0]
## resort type  unique code

data['resort_unique_code'] = make_identifier(data[['state_code_resort','cluster_code','resort_region_code','resort_type_code']])



## member unique code

data['member_unique_code'] = make_identifier(data[['state_code_residence','persontravellingid', 'member_age_buckets']])



##Resort product unoque code

data['resort_product_unique_code'] = make_identifier(data[['resort_region_code','resort_type_code','main_product_code']])



##room product unique code

data['roomtype_product_unique_code'] = make_identifier(data[['resort_region_code','resort_type_code','room_type_booked_code',

                                                             'main_product_code']])



## unique channel booking code

data['channel_unique_code'] = make_identifier(data[['channel_code','booking_type_code']])
## lets see the data once more

data.head()
data.shape, train.shape[0]+test.shape[0]
## add date information columns for each datetype. Although all are not essential but add them anyway

add_datepart(data, 'booking_date')

add_datepart(data, 'checkin_date')

add_datepart(data, 'checkout_date')

#data = data.drop(['checkout_date', 'checkin_date'],axis = 1)
## label encode the object columns

## we are not encoding reservationn id as these will not be present in the model building

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in [ 'member_age_buckets', 'cluster_code', 'reservationstatusid_code','over_stay','season_holidayed_code_na',

           'state_code_residence_na','memberid','resort_id', 'advance_booking_group','total_pax_differ_group','stay_over_group']:

    data[col] = le.fit_transform(data[col])
## convert these below object columns to category as model  handles this type much better

# for col in [ 'member_age_buckets', 'cluster_code', 'reservationstatusid_code','over_stay','season_holidayed_code_na',

#            'state_code_residence_na','memberid','resort_id']:

#     data[col] = data[col].astype('category')

# data['advance_booking_group'] = le.fit_transform(data['advance_booking_group'])
data.shape
data.info()
## everything looks good. only 2 object columns which we ought to remove them next
## split the data to train and test set based on the type level

## define y (target) column

train = data[data['type'] == 'train']

test = data[data['type'] == 'test']

y = train['amount_spent_per_room_night_scaled']



## delete the  id columns which are not part of the model

train_id = train.reservation_id

test_id = test.reservation_id

train_dl = train.drop(['type','reservation_id'], axis = 1)

train = train.drop(['type','amount_spent_per_room_night_scaled','reservation_id'], axis = 1)

test = test.drop(['type','amount_spent_per_room_night_scaled','reservation_id'], axis = 1)

train.shape, y.shape,test.shape, train.shape[0]+test.shape[0]
## split the dataset to train and valid.

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.25, random_state=2)
max(y), min(y)
## define rmse as per problem statement

def rmse(x,y): return math.sqrt(((x-y)**2).mean()) * 100
from lightgbm import LGBMRegressor

# lgb = LGBMRegressor(random_state=1, n_jobs=-1,n_estimators=300, num_leaves=15, learning_rate=0.1,min_child_samples=3, 

#                     reg_alpha=0.0, reg_lambda=0.0, importance_type='gain')

# lgb.fit(X_train, y_train)

# y_valid_pred = lgb.predict(X_valid)

# rmse(y_valid, y_valid_pred)
##97.82

# feature_imp = pd.DataFrame(sorted(zip(lgb.feature_importances_,X_train.columns)), columns=['Value','Feature'])

# feature_imp.sort_values('Value',ascending=False)[0:10]
# to_keep = feature_imp[feature_imp.Value>100].Feature

# df_keep = train[to_keep].copy()

# X_train, X_valid, y_train, y_valid = train_test_split(df_keep, y, test_size=0.25, random_state=2)
from lightgbm import LGBMRegressor

from sklearn.model_selection import StratifiedKFold, KFold

## k-fold model define

def k_folds_lgb(X, y, X_test, k,n_est,num_leaves,lr):

    folds = KFold( n_splits = k, shuffle=True, random_state=2)

    y_test = 0

    y_oof = np.zeros((X.shape[0]))

    score = 0

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

        clf =  LGBMRegressor(random_state =1, n_estimators = n_est, n_jobs = -1,learning_rate=lr, num_leaves=num_leaves,reg_alpha=0.0, 

                             reg_lambda=0.0, min_child_samples=3 ) #categorical_feature = [13,15,16,17,18,19,20,26,27]

        clf.fit(X.iloc[train_idx], y[train_idx])

        y_oof[val_idx] = np.clip(clf.predict(X.iloc[val_idx]),0,20)

        y_test += np.clip(clf.predict(X_test),0,20) / folds.n_splits

        score += rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])

        print('Fold: {} score: {}'.format(i,rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])))

    print('Avg 100* RMSE', score / folds.n_splits) 

        

    return y_oof, y_test 
y_oof, y_test_lgb = k_folds_lgb(train, y, test, k= 30,n_est=300,num_leaves = 15, lr = 0.10)

min(y_test_lgb), max(y_test_lgb)

#98.08
#lowest #Avg RMSE 97.72307084242247 for k=50,lr=0.1,num_leaves=15,n_est=300,reg_alpha=0.1,min_chid_sample=3

submit = pd.DataFrame({'reservation_id':test_id, 'amount_spent_per_room_night_scaled':y_test_lgb}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

#submit['num_orders'][submit['amount_spent_per_room_night_scaled'] <0] = 0

submit.to_csv('submit_klgb.csv', index=False)

create_download_link('submit_klgb.csv')
submit.head()
from sklearn.ensemble import AdaBoostRegressor

## k-fold model define

def k_folds_ada(X, y, X_test, k,n_est,lr):

    folds = KFold( n_splits = k, shuffle=True, random_state=2)

    y_test = 0

    y_oof = np.zeros((X.shape[0]))

    score = 0

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

        clf =  AdaBoostRegressor(random_state =1, n_estimators = n_est, learning_rate=lr, loss = 'exponential' ) 

        clf.fit(X.iloc[train_idx], y[train_idx])

        y_oof[val_idx] = np.clip(clf.predict(X.iloc[val_idx]),0,20)

        y_test += np.clip(clf.predict(X_test),0,20) / folds.n_splits

        score += rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])

        print('Fold: {} score: {}'.format(i,rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])))

    print('Avg 100* RMSE', score / folds.n_splits) 

        

    return y_oof, y_test 
# y_oof, y_test_ada = k_folds_ada(train, y, test, k= 5,n_est=10, lr = 0.20)

# min(y_test_ada), max(y_test_ada)

# #103.4
# #

# submit = pd.DataFrame({'reservation_id':test_id, 'amount_spent_per_room_night_scaled':y_test_ada}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

# #submit['num_orders'][submit['amount_spent_per_room_night_scaled'] <0] = 0

# submit.to_csv('submit_kada.csv', index=False)

# create_download_link('submit_kada.csv')
from sklearn.ensemble import GradientBoostingRegressor

# gbm = GradientBoostingRegressor(random_state =1, n_estimators = 50,learning_rate=0.25,max_depth=5)                              

# gbm.fit(X_train, y_train)

# y_valid_pred = gbm.predict(X_valid)

# rmse(y_valid_pred, y_valid)

# ##98.2395181080143
## k-fold model define

def k_folds_gbm(X, y, X_test, k,n_est, lr):

    folds = KFold( n_splits = k, shuffle=True, random_state=2)

    y_test = 0

    y_oof = np.zeros((X.shape[0]))

    score = 0

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

        clf =  GradientBoostingRegressor(n_estimators = n_est, learning_rate = lr, random_state=1, max_depth=5)

        clf.fit(X.iloc[train_idx], y[train_idx])

        y_oof[val_idx] = np.clip(clf.predict(X.iloc[val_idx]),0,20)

        y_test += np.clip(clf.predict(X_test),0,20) / folds.n_splits

        score += rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])

        print('Fold: {} score: {}'.format(i,rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])))

    print('Avg 100* RMSE', score / folds.n_splits) 

        

    return y_oof, y_test 
y_oof, y_test_gbm = k_folds_gbm(train, y, test, k= 20,n_est=100, lr = 0.25)

min(y_test_gbm), max(y_test_gbm)
# submit = pd.DataFrame({'reservation_id':test_id, 'amount_spent_per_room_night_scaled':y_test_gbm}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

# #submit['num_orders'][submit['amount_spent_per_room_night_scaled'] <0] = 0

# submit.to_csv('submit_kgbm.csv', index=False)

# create_download_link('submit_kgbm.csv')
# submit.head()
from xgboost import XGBRegressor

# xgb = XGBRegressor(n_estimators = 50, n_jobs = -1,learning_rate = 0.1, gamma = 0.00, random_state=1,

#                            colsample_bytree=1, max_depth=7)

# xgb.fit(X_train, y_train)

# y_valid_pred = xgb.predict(X_valid)

# rmse(y_valid_pred, y_valid)

# ### 98.52735932116437
from xgboost import XGBRegressor

## k-fold model define

def k_folds_xgb(X, y, X_test, k,n_est, lr):

    folds = KFold( n_splits = k, shuffle=True, random_state=2)

    y_test = 0

    y_oof = np.zeros((X.shape[0]))

    score = 0

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

        clf =  XGBRegressor(n_estimators = n_est, n_jobs = -1,learning_rate = lr, gamma = 0.00, random_state=1,

                           colsample_bytree=1, max_depth=7)

        clf.fit(X.iloc[train_idx], y[train_idx])

        y_oof[val_idx] = np.clip(clf.predict(X.iloc[val_idx]),0,20)

        y_test += np.clip(clf.predict(X_test),0,20) / folds.n_splits

        score += rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])

        print('Fold: {} score: {}'.format(i,rmse(np.clip(clf.predict(X.iloc[val_idx]),0,20), y[val_idx])))

    print('Avg 100* RMSE', score / folds.n_splits) 

        

    return y_oof, y_test 
# y_oof, y_test_xgb = k_folds_xgb(train, y, test, k= 5,n_est=30, lr = 0.10)

# min(y_test_xgb), max(y_test_xgb)
# submit = pd.DataFrame({'reservation_id':test_id, 'amount_spent_per_room_night_scaled':y_test_xgb}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

# #submit['num_orders'][submit['amount_spent_per_room_night_scaled'] <0] = 0

# submit.to_csv('submit_kxgb.csv', index=False)

# create_download_link('submit_kxgb.csv')
submit.head()
from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(random_state =1, n_estimators = 50, n_jobs = -1,min_samples_leaf=15,max_features = 'sqrt')

# rf.fit(X_train, y_train)

# y_valid_pred = rf.predict(X_valid)

# rmse(y_valid_pred, y_valid)

### 99.29315280701118
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import StratifiedKFold, KFold

## k-fold model define

def k_folds_rf(X, y, X_test, k,n_est,num_leaves,max_feat):

    folds = KFold( n_splits = k, shuffle=True, random_state=2)

    y_test = 0

    y_oof = np.zeros((X.shape[0]))

    score = 0

    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):

        clf =  RandomForestRegressor(random_state =1, n_estimators = n_est, n_jobs = -1,min_samples_leaf=num_leaves,max_features = max_feat)

        clf.fit(X.iloc[train_idx], y[train_idx])

        y_oof[val_idx] = clf.predict(X.iloc[val_idx])

        y_test += clf.predict(X_test) / folds.n_splits

        score += rmse(clf.predict(X.iloc[val_idx]), y[val_idx])

        print('Fold: {} score: {}'.format(i,rmse(clf.predict(X.iloc[val_idx]), y[val_idx])))

    print('Avg 100* RMSE', score / folds.n_splits) 

        

    return y_oof, y_test 
y_oof, y_test_rf = k_folds_rf(train, y, test, k= 20,n_est=150,num_leaves = 15, max_feat ='sqrt')

min(y_test_rf), max(y_test_rf)
##avg score 99.16587677246167

submit = pd.DataFrame({'reservation_id':test_id, 'amount_spent_per_room_night_scaled':y_test_rf}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

#submit['num_orders'][submit['amount_spent_per_room_night_scaled'] <0] = 0

submit.to_csv('submit_krf.csv', index=False)

create_download_link('submit_krf.csv')
submit.head()
## LGBM performed better, so giving modre weights

#y_test_avg = 0.27 * y_test_lgb + 0.24 * y_test_xgb + 0.25 * y_test_gbm + 0.24 * y_test_rf

y_test_avg = 0.35 * y_test_lgb  + 0.33 * y_test_gbm + 0.32 * y_test_rf

submit = pd.DataFrame({'reservation_id':test_id, 'amount_spent_per_room_night_scaled':y_test_avg}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

#submit['num_orders'][submit['amount_spent_per_room_night_scaled'] <0] = 0

submit.to_csv('submit_avg.csv', index=False)

create_download_link('submit_avg.csv')
submit.head()
path = '../working'

!ls {path}
from argparse import Namespace

#Hyperparameters

s= Namespace( **{

    "l1":4400,

    "l2":2300,

    "ps1":0.25,

    "ps2":0.15,

    "emb_drop":0.14,

    "batchsize":64,

    "lrate":0.006,

    "lrate_ratio":9,

    "wd":0.17,

    "l1epoch":4,

    "l2epoch":2,

    "l3epoch":2,

})
#train_dl.info()
from random import sample

valid_idx = train_dl.sample(frac=0.01, random_state=1).index.tolist()
dep_var = 'amount_spent_per_room_night_scaled'

cat_vars = ['channel_code', 'main_product_code','numberofadults','numberofchildren','persontravellingid','resort_region_code',

           'resort_type_code','room_type_booked_code','season_holidayed_code','state_code_residence','state_code_resort','member_age_buckets',

           'booking_type_code','memberid','cluster_code','reservationstatusid_code','resort_id','season_holidayed_code_na','over_stay',

           'state_code_residence_na','advance_booking_group','booking_Year','booking_Month','booking_Week','booking_Day','booking_Dayofweek',

           'booking_Dayofyear','booking_Is_month_end','booking_Is_month_start','booking_Is_quarter_end','booking_Is_quarter_start',

           'booking_Is_year_end','booking_Is_year_start','booking_Elapsed','checkin_Year', 'checkin_Month', 'checkin_Week', 'checkin_Day',

            'checkin_Dayofweek', 'checkin_Dayofyear', 'checkin_Is_month_end','checkin_Is_month_start', 'checkin_Is_quarter_end', 

            'checkin_Is_quarter_start', 'checkin_Is_year_end','checkin_Is_year_start', 'checkin_Elapsed', 'checkout_Year',

            'checkout_Month', 'checkout_Week', 'checkout_Day', 'checkout_Dayofweek','checkout_Dayofyear', 'checkout_Is_month_end',

            'checkout_Is_month_start', 'checkout_Is_quarter_end', 'checkout_Is_quarter_start', 'checkout_Is_year_end',

            'checkout_Is_year_start', 'checkout_Elapsed']

cont_vars = ['roomnights','total_pax','total_people','total_adult_night','total_people_night','days_advance','actual_stay']

procs=[FillMissing, Categorify, Normalize]
databunch = (TabularList.from_df(train_dl, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)

                .split_by_idx(valid_idx)

                .label_from_df(cols=dep_var, label_cls=FloatList, log=False)

                .add_test(TabularList.from_df(test, path=path, cat_names=cat_vars, cont_names=cont_vars))

                .databunch())

databunch.batch_size=s.batchsize
databunch.show_batch(rows=5)
max_y = np.max(train_dl['amount_spent_per_room_night_scaled'])*1.2

y_range = torch.tensor([0, max_y], device=defaults.device)
learn = tabular_learner(databunch, layers=[s.l1,s.l2], ps=[s.ps1,s.ps2], emb_drop=s.emb_drop, y_range=y_range, metrics=rmse)
learn.model

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(s.l1epoch, s.lrate, wd=s.wd)
learn.recorder.plot_losses()
learn.fit_one_cycle(s.l2epoch, s.lrate/s.lrate_ratio, wd=s.wd)
learn.fit_one_cycle(s.l3epoch, s.lrate/(s.lrate_ratio*s.lrate_ratio), wd=s.wd)
test_preds = learn.get_preds(DatasetType.Test)

y_test_preds = (test_preds[0].data).numpy().T[0]

submit = pd.DataFrame({'reservation_id':test_id, 'amount_spent_per_room_night_scaled':y_test_preds}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

#submit['num_orders'][submit['amount_spent_per_room_night_scaled'] <0] = 0

submit.to_csv('submit_dl.csv', index=False)

create_download_link('submit_dl.csv')
submit.head()