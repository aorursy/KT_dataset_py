import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/train.csv')

test = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/test.csv')

users = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/users.csv')

samp_sub = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')



print('train:', train.shape)

print('test:', test.shape)

print('users:', users.shape)

print('sample submission:', samp_sub.shape)
train
test
users
samp_sub
train['open_flag'].value_counts()
train = pd.merge(train, users, on = 'user_id', how = 'left')

test = pd.merge(test, users, on = 'user_id', how = 'left')
train.info()
# change type of categorical columns to str

train['country_code'] = train['country_code'].astype(str)

train['attr_1'] = train['attr_1'].astype(str)

train['attr_2'] = train['attr_2'].astype(str)

train['attr_3'] = train['attr_3'].astype(str)



test['country_code'] = test['country_code'].astype(str)

test['attr_1'] = test['attr_1'].astype(str)

test['attr_2'] = test['attr_2'].astype(str)

test['attr_3'] = test['attr_3'].astype(str)
import datetime as dt



train['grass_date'] = pd.to_datetime(train['grass_date'])

test['grass_date'] = pd.to_datetime(test['grass_date'])
train_flag0 = train[train['open_flag'] == 0]

train_flag1 = train[train['open_flag'] == 1]
train_flag1['country_code'].hist()
train_flag0['country_code'].hist()
print(train['grass_date'].min(), train['grass_date'].max())

print(test['grass_date'].min(), test['grass_date'].max())
# add new column 'day_of_week'

train['day_of_week'] = train['grass_date'].dt.day_name()

test['day_of_week'] = test['grass_date'].dt.day_name()



train['day_of_month'] = train['grass_date'].dt.day

test['day_of_month'] = test['grass_date'].dt.day
train['day_of_week'].value_counts()
print(train['subject_line_length'].min(), test['subject_line_length'].max())
train_flag1['subject_line_length'].hist()
train_flag0['subject_line_length'].hist()
last_open_day_train = train[train['last_open_day'] != 'Never open'].last_open_day.astype(int)

last_open_day_test = test[test['last_open_day'] != 'Never open'].last_open_day.astype(int)

print(last_open_day_train.min(), last_open_day_train.max())

print(last_open_day_test.min(), last_open_day_test.max())
train[(train['last_open_day'] == 'Never open') & (train['open_flag'] == 1)]
train_temp1 = train[(train['last_open_day'] != 'Never open') & (train['open_flag'] == 1)]

train_temp0 = train[(train['last_open_day'] != 'Never open') & (train['open_flag'] == 0)]

train_temp1['last_open_day'] = train_temp1['last_open_day'].astype(int)

train_temp0['last_open_day'] = train_temp0['last_open_day'].astype(int)
train_temp1['last_open_day'].hist()
train_temp1 = train_temp1[(train_temp1['last_open_day'] < 300)]

train_temp1['last_open_day'].hist()
train_temp0['last_open_day'].hist()
train_temp0[(train_temp0['last_open_day'] < 500)]['last_open_day'].hist()
test[(test['last_open_day'] == 'Never open')]
last_login_day_train = train[train['last_login_day'] != 'Never login'].last_login_day.astype(int)

last_login_day_test = test[test['last_login_day'] != 'Never login'].last_login_day.astype(int)

print(last_login_day_train.min(), last_login_day_train.max())

print(last_login_day_test.min(), last_login_day_test.max())
train_temp1 = train[(train['last_login_day'] != 'Never login') & (train['open_flag'] == 1)]

train_temp0 = train[(train['last_login_day'] != 'Never login') & (train['open_flag'] == 1)]

train_temp1['last_login_day'] = train_temp1['last_login_day'].astype(int)

train_temp0['last_login_day'] = train_temp0['last_login_day'].astype(int)
train_temp1['last_login_day'].hist()
train_temp1[train_temp1['last_login_day'] < 200]['last_login_day'].hist()
train_temp0[train_temp0['last_login_day'] < 200]['last_login_day'].hist()
train_temp1[train_temp1['last_login_day'] > 15000]
len(test[test['last_login_day'] == 'Never login'])
train['last_checkout_day'].value_counts()
last_checkout_day_train = train[train['last_checkout_day'] != 'Never checkout'].last_checkout_day.astype(int)

last_checkout_day_test = test[test['last_checkout_day'] != 'Never checkout'].last_checkout_day.astype(int)

print(last_checkout_day_train.min(), last_checkout_day_train.max())

print(last_checkout_day_test.min(), last_checkout_day_test.max())
train[(train['last_checkout_day'] == 'Never checkout') & (train['open_flag'] == 1)]
train_temp = train[(train['last_checkout_day'] != 'Never checkout') & (train['open_flag'] == 1)]

train_temp['last_checkout_day'] = train_temp['last_checkout_day'].astype(int)
train_temp['last_checkout_day'].hist()
train_temp = train_temp[train_temp['last_checkout_day'] < 100]

train_temp['last_checkout_day'].hist()
train_temp = train[(train['last_checkout_day'] != 'Never checkout') & (train['open_flag'] == 0)]

train_temp['last_checkout_day'] = train_temp['last_checkout_day'].astype(int)
train_temp['last_checkout_day'].hist()
train_temp = train_temp[train_temp['last_checkout_day'] < 100]

train_temp['last_checkout_day'].hist()
train['open_count_last_10_days'].value_counts()
train_flag1 = train[train['open_flag'] == 1]

train_flag0 = train[train['open_flag'] == 0]
train_flag1['open_count_last_10_days'].hist()
train_flag0['open_count_last_10_days'].hist()
train_flag1['open_count_last_30_days'].hist()
train_flag0['open_count_last_30_days'].hist()
train_flag1['open_count_last_60_days'].hist()
train_flag0['open_count_last_60_days'].hist()
train_flag1['login_count_last_10_days'].hist()
train_flag0['login_count_last_10_days'].hist(bins = [0, 20, 40, 60, 80, 100])
train_flag1['login_count_last_30_days'].hist()
train_flag0['login_count_last_30_days'].hist(bins = [0, 100, 200, 300])
train_flag1['login_count_last_60_days'].hist()
train_flag0['login_count_last_60_days'].hist()
train_flag1['checkout_count_last_10_days'].hist()
train_flag0['checkout_count_last_10_days'].hist()
train_flag1[train_flag1['checkout_count_last_30_days'] < 100]['checkout_count_last_30_days'].hist()
train_flag0[train_flag0['checkout_count_last_30_days'] < 100]['checkout_count_last_30_days'].hist()
train_flag1[train_flag1['checkout_count_last_60_days'] < 100]['checkout_count_last_60_days'].hist()
train_flag0[train_flag0['checkout_count_last_60_days'] < 100]['checkout_count_last_60_days'].hist()
train.describe()
test.describe()
train_flag1 = train[train['open_flag'] == 1]

train_flag0 = train[train['open_flag'] == 0]
train_flag1['age'].hist()
train_flag0['age'].hist()
train_flag1[train_flag1['age'] > 110]['age'].hist()
train_flag0[train_flag0['age'] > 110]['age'].hist()
train[(train.age > 116)]
train['age_class'] = train['age'].isna()

train['age_class'] = train['age_class'].map({True:'Unknown',False:'<>'})



test['age_class'] = test['age'].isna()

test['age_class'] = test['age_class'].map({True:'Unknown',False:'<>'})



train.loc[train['age'] >= 30, 'age_class'] = '>=30'

train.loc[train['age'] < 30, 'age_class'] = '<30'



test.loc[test['age'] >= 30, 'age_class'] = '>=30'

test.loc[test['age'] < 30, 'age_class'] = '<30'
train.loc[train['age'] > 110, 'age'] = np.nan

test.loc[test['age'] > 110, 'age'] = np.nan
train.loc[train['age'] < 0, 'age'] = np.nan

test.loc[test['age'] < 0, 'age'] = np.nan
train_flag1['domain'].value_counts()
train_flag0['domain'].value_counts()
train_flag1['domain'].hist()
train_flag0['domain'].hist()
list_low_domain = ['@163.com','@gmail.com','@yahoo.com','@ymail.com'] # low open rate

list_med_domain = ['@outlook.com','@qq.com','@rocketmail.com'] # medium open rate

list_high_domain = ['@hotmail.com','@icloud.com','@live.com','other'] # high open rate



def make_domain_type(dom) :

    if dom in list_low_domain :

        res = 'low_domain'

    elif dom in list_med_domain :

        res = 'med_domain'

    elif dom in list_high_domain :

        res = 'high_domain'

        

    return res



train['domain_type'] = train.apply(lambda x : make_domain_type(x['domain']), axis=1)

test['domain_type'] = test.apply(lambda x : make_domain_type(x['domain']), axis=1)
train_flag1['attr_1'].value_counts()
train_flag0['attr_1'].value_counts()
train[(train['open_flag'] == 1) & (train['attr_1'] == 'nan')]
train[(train['open_flag'] == 0) & (train['attr_1'] == 'nan')]
train_flag1['attr_2'].value_counts()
train_flag0['attr_2'].value_counts()
train_flag1['attr_3'].value_counts()
train_flag0['attr_3'].value_counts()
train.isna().sum()
test.isna().sum()
train[['attr_1', 'attr_2', 'attr_3']]
# add a new unknown variable per attribute

train['attr_1'] = train['attr_1'].replace(['nan'], '2.0')

train['attr_2'] = train['attr_2'].replace(['nan'], '2.0')

test['attr_1'] = test['attr_1'].replace(['nan'], '2.0')

test['attr_2'] = test['attr_2'].replace(['nan'], '2.0')



# print(train['attr_1'].isna().sum(), test['attr_1'].isna().sum())

# print(train['attr_2'].isna().sum(), test['attr_1'].isna().sum())
train['attr_1'].value_counts()
train.isna().sum()
test.isna().sum()
train._get_numeric_data().columns
train['grass_date'] = train['grass_date'].dt.tz_convert(None)

train['grass_date'] = (train['grass_date'] - dt.datetime(1970,1,1)).dt.total_seconds()



test['grass_date'] = test['grass_date'].dt.tz_convert(None)

test['grass_date'] = (test['grass_date'] - dt.datetime(1970,1,1)).dt.total_seconds()
# never_open, boolean

train['never_open'] = train['last_open_day'].apply(lambda x: 'never_open' if x == 'Never open' else 'open')

test['never_open'] = test['last_open_day'].apply(lambda x: 'never_open' if x == 'Never open' else 'open')



# never_login, boolean

train['never_login'] = train['last_login_day'].apply(lambda x: 'never_login' if x == 'Never login' else 'login')

test['never_login'] = test['last_login_day'].apply(lambda x: 'never_login' if x == 'Never login' else 'login')



# never_checkout, boolean

train['never_checkout'] = train['last_checkout_day'].apply(lambda x: 'never_checkout' if x == 'Never checkout' else 'checkout')

test['never_checkout'] = test['last_checkout_day'].apply(lambda x: 'never_checkout' if x == 'Never checkout' else 'checkout')
train['never_open'].value_counts()
train[['last_open_day', 'last_login_day', 'last_checkout_day']]
train['last_open_day'].value_counts()
len(train[train['last_open_day'] == '0'])
train['last_open_day'] = train['last_open_day'].replace(['Never open'], 1000)

train['last_open_day'] = train['last_open_day'].astype(int)

train['last_open_day'].value_counts()
test['last_open_day'] = test['last_open_day'].replace(['Never open'], 1000)

test['last_open_day'] = test['last_open_day'].astype(int)

test['last_open_day'].value_counts()
train['last_login_day'] = train['last_login_day'].replace(['Never login'], 19000)

train['last_login_day'] = train['last_login_day'].astype(int)

train['last_login_day'].value_counts()
test['last_login_day'] = test['last_login_day'].replace(['Never login'], 19000)

test['last_login_day'] = test['last_login_day'].astype(int)

test['last_login_day'].value_counts()
train['last_checkout_day'].value_counts()
len(train[train['last_checkout_day'] == '0'])
train['last_checkout_day'] = train['last_checkout_day'].replace(['Never checkout'], 1500)

train['last_checkout_day'] = train['last_checkout_day'].astype(int)

train['last_checkout_day'].value_counts()
test['last_checkout_day'] = test['last_checkout_day'].replace(['Never checkout'], 1500)

test['last_checkout_day'] = test['last_checkout_day'].astype(int)

test['last_checkout_day'].value_counts()
train.info()
train_feat = train._get_numeric_data()

train_feat
test_feat = test._get_numeric_data()
dom_flag = pd.get_dummies(train['domain'])

train_feat = pd.concat([train_feat, dom_flag], axis = 1)

train_feat
dom_flag
dom_flag_test = pd.get_dummies(test['domain'])

test_feat = pd.concat([test_feat, dom_flag_test], axis = 1)

test_feat
ccode_flag = pd.get_dummies(train['country_code'])

train_feat = pd.concat([train_feat, ccode_flag], axis = 1)

train_feat
ccode_flag_test = pd.get_dummies(test['country_code'])

test_feat = pd.concat([test_feat, ccode_flag_test], axis = 1)

test_feat
dweek_flag = pd.get_dummies(train['day_of_week'])

train_feat = pd.concat([train_feat, dweek_flag], axis = 1)

train_feat
dweek_flag_test = pd.get_dummies(test['day_of_week'])

test_feat = pd.concat([test_feat, dweek_flag_test], axis = 1)

test_feat
open_flag = pd.get_dummies(train['never_open'])

train_feat = pd.concat([train_feat, open_flag], axis = 1)

print(train_feat.shape)



open_flag_test = pd.get_dummies(test['never_open'])

test_feat = pd.concat([test_feat, open_flag_test], axis = 1)

print(test_feat.shape)
login_flag = pd.get_dummies(train['never_login'])

train_feat = pd.concat([train_feat, login_flag], axis = 1)

print(train_feat.shape)



login_flag_test = pd.get_dummies(test['never_login'])

test_feat = pd.concat([test_feat, login_flag_test], axis = 1)

print(test_feat.shape)
checkout_flag = pd.get_dummies(train['never_checkout'])

train_feat = pd.concat([train_feat, checkout_flag], axis = 1)

print(train_feat.shape)



checkout_flag_test = pd.get_dummies(test['never_checkout'])

test_feat = pd.concat([test_feat, checkout_flag_test], axis = 1)

print(test_feat.shape)
attr1_mapper_encode = {'0.0': 'attr_10',

                       '1.0': 'attr_11',

                       '2.0': 'attr_12'}



train['attr_1'] = train['attr_1'].map(attr1_mapper_encode)

test['attr_1'] = test['attr_1'].map(attr1_mapper_encode)
attr2_mapper_encode = {'0.0': 'attr_20',

                       '1.0': 'attr_21',

                       '2.0': 'attr_22'}



train['attr_2'] = train['attr_2'].map(attr2_mapper_encode)

test['attr_2'] = test['attr_2'].map(attr2_mapper_encode)
attr3_mapper_encode = {'0.0': 'attr_30',

                       '1.0': 'attr_31',

                       '2.0': 'attr_32',

                       '3.0': 'attr_33', 

                       '4.0': 'attr_34'}



train['attr_3'] = train['attr_3'].map(attr3_mapper_encode)

test['attr_3'] = test['attr_3'].map(attr3_mapper_encode)
attr1_flag = pd.get_dummies(train['attr_1'])

train_feat = pd.concat([train_feat, attr1_flag], axis = 1)

print(train_feat.shape)



attr1_flag_test = pd.get_dummies(test['attr_1'])

test_feat = pd.concat([test_feat, attr1_flag_test], axis = 1)

print(test_feat.shape)
attr2_flag = pd.get_dummies(train['attr_2'])

train_feat = pd.concat([train_feat, attr2_flag], axis = 1)

print(train_feat.shape)



attr2_flag_test = pd.get_dummies(test['attr_2'])

test_feat = pd.concat([test_feat, attr2_flag_test], axis = 1)

print(test_feat.shape)
attr3_flag = pd.get_dummies(train['attr_3'])

train_feat = pd.concat([train_feat, attr3_flag], axis = 1)

print(train_feat.shape)



attr3_flag_test = pd.get_dummies(test['attr_3'])

test_feat = pd.concat([test_feat, attr3_flag_test], axis = 1)

print(test_feat.shape)
train.dtypes
train_feat
test_feat
features = [c for c in train_feat.columns if c not in ['open_flag', 'user_id', 'row_id', 'age_class', 'domain_type', 'day_of_month']]

len(features)
!pip install impyute
print(list(train_feat[features].columns).index('age'), list(test_feat[features].columns).index('age'))
from impyute.imputation.cs import mice



train_imputed = mice(train_feat[features].values)

mice_ages = train_imputed[:, list(train_feat[features].columns).index('age')]

train_feat['age'] = mice_ages



test_imputed = mice(test_feat[features].values)

mice_ages_test = test_imputed[:, list(test_feat[features].columns).index('age')]

test_feat['age'] = mice_ages_test
train['age'].describe()
train_feat['age'].describe()
test['age'].describe()
test_feat['age'].describe()
train['20_interval'] = train['open_count_last_30_days'] - train['open_count_last_10_days']

train['30_interval'] = train['open_count_last_60_days'] - train['open_count_last_30_days']

train['50_interval'] = train['open_count_last_60_days'] - train['open_count_last_10_days']



test['20_interval'] = test['open_count_last_30_days'] - test['open_count_last_10_days']

test['30_interval'] = test['open_count_last_60_days'] - test['open_count_last_30_days']

test['50_interval'] = test['open_count_last_60_days'] - test['open_count_last_10_days']
train['age'] = train_feat['age']

test['age'] = test_feat['age']



train['domain'] = train['domain'].astype('category')

test['domain'] = test['domain'].astype('category')



train['country_code'] = train['country_code'].astype('category')

test['country_code'] = test['country_code'].astype('category')



train['day_of_week'] = train['day_of_week'].astype('category')

test['day_of_week'] = test['day_of_week'].astype('category')



train['attr_1'] = train['attr_1'].astype('category')

test['attr_1'] = test['attr_1'].astype('category')



train['attr_2'] = train['attr_2'].astype('category')

test['attr_2'] = test['attr_2'].astype('category')



train['attr_3'] = train['attr_3'].astype('category')

test['attr_3'] = test['attr_3'].astype('category')



train['never_open'] = train['never_open'].astype('category')

test['never_open'] = test['never_open'].astype('category')



train['never_login'] = train['never_login'].astype('category')

test['never_login'] = test['never_login'].astype('category')



train['never_checkout'] = train['never_checkout'].astype('category')

test['never_checkout'] = test['never_checkout'].astype('category')



train['age_class'] = train['age_class'].astype('category')

test['age_class'] = test['age_class'].astype('category')



train['domain_type'] = train['domain_type'].astype('category')

test['domain_type'] = test['domain_type'].astype('category')
train.info()
train.describe()
train.to_csv('train_feat.csv', index = False)

test.to_csv('test_feat.csv', index = False)
# train_feat = pd.read_csv('../input/shopee-code-league-2020-marketing-analytics/train_feat.csv')

# test_feat = pd.read_csv('../input/shopee-code-league-2020-marketing-analytics/test_feat.csv')

# print(train_feat.shape, test_feat.shape)
# label = train_feat['open_flag']
# samp_sub = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')

# print(samp_sub.shape)
train_feat_copy = train.copy()

test_feat_copy = test.copy()
features = [c for c in train.columns if c not in ['open_flag', 'user_id', 'row_id']]

feat_label = [c for c in train.columns if c not in ['user_id', 'row_id']]

label = train['open_flag']
import h2o

h2o.init()
X = features

Y = 'open_flag'





list_col = X + [Y]
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(train[feat_label], stratify=train['open_flag'], test_size = 0.2, random_state=1111)
h2o_train = h2o.H2OFrame(train_data[list_col])

h2o_val = h2o.H2OFrame(val_data[list_col])

h2o_test = h2o.H2OFrame(test[X])
X_cat = ['country_code', 'domain', 'day_of_week', 'attr_1', 'attr_2', 'attr_3',

         'never_open', 'never_login', 'never_checkout', 'age_class', 'domain_type']



for var in X_cat :

    h2o_train[var] = h2o_train[var].asfactor()

    h2o_val[var] = h2o_val[var].asfactor()

    h2o_test[var] = h2o_test[var].asfactor()

    

h2o_train[Y] = h2o_train[Y].asfactor()

h2o_val[Y] = h2o_val[Y].asfactor()
from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.estimators import H2OXGBoostEstimator

from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

import time



def h2o_compare_models(df_train, df_test, X, Y) :

    

    start = time.time()

    

    # Initialize all model 

    glm = H2OGeneralizedLinearEstimator(family='binomial', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    gbm = H2OGradientBoostingEstimator(distribution='bernoulli', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    xgb = H2OXGBoostEstimator(distribution='bernoulli', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    lgbm = H2OXGBoostEstimator(distribution='bernoulli', tree_method="hist", grow_policy="lossguide",

                              nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    rf = H2ORandomForestEstimator(distribution='bernoulli', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    ext = H2ORandomForestEstimator(distribution='bernoulli', histogram_type="Random",

                                  nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    

    # Train model

    glm.train(x=X, y=Y, training_frame=df_train)

    gbm.train(x=X, y=Y, training_frame=df_train)

    xgb.train(x=X, y=Y, training_frame=df_train)

    lgbm.train(x=X, y=Y, training_frame=df_train)

    rf.train(x=X, y=Y, training_frame=df_train)

    ext.train(x=X, y=Y, training_frame=df_train)

    

    # Calculate train metrics 

    from sklearn.metrics import matthews_corrcoef

    train_glm = matthews_corrcoef(h2o_train[Y].as_data_frame(), glm.predict(h2o_train)['predict'].as_data_frame())

    train_gbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), gbm.predict(h2o_train)['predict'].as_data_frame())

    train_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.predict(h2o_train)['predict'].as_data_frame())

    train_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.predict(h2o_train)['predict'].as_data_frame())

    train_rf = matthews_corrcoef(h2o_train[Y].as_data_frame(), rf.predict(h2o_train)['predict'].as_data_frame())

    train_ext = matthews_corrcoef(h2o_train[Y].as_data_frame(), ext.predict(h2o_train)['predict'].as_data_frame())



    # Calculate CV metrics for all model 

    met_glm = matthews_corrcoef(h2o_train[Y].as_data_frame(), glm.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_gbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), gbm.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_rf = matthews_corrcoef(h2o_train[Y].as_data_frame(), rf.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_ext = matthews_corrcoef(h2o_train[Y].as_data_frame(), ext.cross_validation_holdout_predictions()['predict'].as_data_frame())

    

    # Calculate holdout metrics

    from sklearn.metrics import matthews_corrcoef

    hold_glm = matthews_corrcoef(h2o_val[Y].as_data_frame(), glm.predict(h2o_val)['predict'].as_data_frame())

    hold_gbm = matthews_corrcoef(h2o_val[Y].as_data_frame(), gbm.predict(h2o_val)['predict'].as_data_frame())

    hold_xgb = matthews_corrcoef(h2o_val[Y].as_data_frame(), xgb.predict(h2o_val)['predict'].as_data_frame())

    hold_lgbm = matthews_corrcoef(h2o_val[Y].as_data_frame(), lgbm.predict(h2o_val)['predict'].as_data_frame())

    hold_rf = matthews_corrcoef(h2o_val[Y].as_data_frame(), rf.predict(h2o_val)['predict'].as_data_frame())

    hold_ext = matthews_corrcoef(h2o_val[Y].as_data_frame(), ext.predict(h2o_val)['predict'].as_data_frame())

    

    # Make result dataframe

    result = pd.DataFrame({'Model':['GLM','GBM','XGB','LGBM','RF','ExtraTree'],

                          'Train Metrics':[train_glm,train_gbm,train_xgb,train_lgbm,train_rf,train_ext],

                          'CV Metrics':[met_glm,met_gbm,met_xgb,met_lgbm,met_rf,met_ext],

                          'Holdout Metrics':[hold_glm,hold_gbm,hold_xgb,hold_lgbm,hold_rf,hold_ext]})

    

    end = time.time()

    print('Time Used :',(end-start)/60)

    

    return result.sort_values('Holdout Metrics')
res = h2o_compare_models(h2o_train, h2o_test, X, Y) 

res
from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.estimators import H2OXGBoostEstimator

from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

import time
from h2o.estimators import H2OXGBoostEstimator

from h2o.grid.grid_search import H2OGridSearch

from sklearn.metrics import log_loss



start = time.time()



xgb = H2OXGBoostEstimator(distribution='bernoulli', nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo')



xgb_params = {'max_depth' : [7,9,11], # picked after max depth search

                'sample_rate': [x/100. for x in range(20,101)],

                'col_sample_rate' : [x/100. for x in range(20,101)],

                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],

                'min_split_improvement': [0,1e-8,1e-6,1e-4],

              'reg_lambda':list(np.arange(0.5,1.05,0.05)),

              'reg_alpha':list(np.arange(0.01,0.11,0.01)),

             'learn_rate':list(np.arange(0.01,0.11,0.01)),

             'booster':['dart','gbtree']}



# Search criteria

search_criteria = {'strategy': "RandomDiscrete",

                   'max_runtime_secs': 3600,

                   'max_models': 20,

                   'seed' : 11,

                   'stopping_rounds' : 5,

                   'stopping_metric' : "auc",

                   'stopping_tolerance': 1e-3

                   }



# Make grid model

xgb_grid = H2OGridSearch(model=xgb,

                          grid_id='best_xgb_cmon',

                          hyper_params=xgb_params,

                          search_criteria=search_criteria)



# Train model

xgb_grid.train(x=X, y=Y, training_frame=h2o_train, validation_frame=h2o_val)



# Get best GLM

xgb_res = xgb_grid.get_grid(sort_by='auc', decreasing=True)

best_xgb = xgb_res.models[0]
from sklearn.metrics import matthews_corrcoef

train_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), best_xgb.predict(h2o_train)['predict'].as_data_frame())

met_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), best_xgb.cross_validation_holdout_predictions()['predict'].as_data_frame())

hold_xgb = matthews_corrcoef(h2o_val[Y].as_data_frame(), best_xgb.predict(h2o_val)['predict'].as_data_frame())



# Print result

print('Train metrics :',train_xgb)

print('CV metrics :',met_xgb)

print('Holdout metrics :',hold_xgb)



end = time.time()

print('Time Used :',(end-start)/60)
pred = best_xgb.predict(h2o_test)['predict'].as_data_frame()

sub = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')

sub['open_flag'] = pred



sub.to_csv('submission_xgb.csv', index=False)
sub['open_flag'].value_counts()
# Tune Model - LGBM - RandomGridSearch

from h2o.estimators import H2OXGBoostEstimator

from h2o.grid.grid_search import H2OGridSearch

from sklearn.metrics import log_loss



start = time.time()



lgbm = H2OXGBoostEstimator(distribution='bernoulli', tree_method="hist", grow_policy="lossguide",

                           nfolds=10, keep_cross_validation_predictions=True, fold_assignment='Modulo',

                           ntrees=100, seed=11, score_tree_interval = 10,

                           stopping_rounds = 5, stopping_metric = "AUC", stopping_tolerance = 1e-4)



# LGBM Params

lgbm_params = {'max_depth' : [7,9,11],  # picked after max depth search

                'sample_rate': [x/100. for x in range(20,101)],

                'col_sample_rate' : [x/100. for x in range(20,101)],

                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],

                'min_split_improvement': [0,1e-8,1e-6,1e-4],

              'reg_lambda':list(np.arange(0.5,1.05,0.05)),

              'reg_alpha':list(np.arange(0.01,0.11,0.01)),

             'learn_rate':list(np.arange(0.01,0.11,0.01)),

             'booster':['dart','gbtree']}



# Search criteria

search_criteria = {'strategy': "RandomDiscrete",

                   'max_runtime_secs': 3600,  # limit the runtime to 60 minutes

                   'max_models': 20,  # build no more than 100 models

                   'seed' : 11,

                   'stopping_rounds' : 5,

                   'stopping_metric' : "auc",

                   'stopping_tolerance': 1e-3

                   }



# Make grid model

lgbm_grid = H2OGridSearch(model=lgbm,

                          grid_id='best_lgbm_cmon',

                          hyper_params=lgbm_params,

                          search_criteria=search_criteria)



# Train model

lgbm_grid.train(x=X, y=Y, training_frame=h2o_train, validation_frame=h2o_val)



# Get best GLM

lgbm_res = lgbm_grid.get_grid(sort_by='auc', decreasing=True)

best_lgbm = lgbm_res.models[0]
from sklearn.metrics import matthews_corrcoef

train_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), best_lgbm.predict(h2o_train)['predict'].as_data_frame())

met_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), best_lgbm.cross_validation_holdout_predictions()['predict'].as_data_frame())

hold_lgbm = matthews_corrcoef(h2o_val[Y].as_data_frame(), best_lgbm.predict(h2o_val)['predict'].as_data_frame())



# Print result

print('Train metrics :',train_lgbm)

print('CV metrics :',met_lgbm)

print('Holdout metrics :',hold_lgbm)



end = time.time()

print('Time Used :',(end-start)/60)
pred = best_lgbm.predict(h2o_test)['predict'].as_data_frame()

sub = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')

sub['open_flag'] = pred



sub.to_csv('submission_lgbm.csv', index=False)
sub['open_flag'].value_counts()