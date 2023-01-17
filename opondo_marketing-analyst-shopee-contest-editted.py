# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/train.csv')

test = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/test.csv')

users = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/users.csv')



print("Train:", train.shape)

print("Test:", test.shape)

print('Users:', users.shape)
df = pd.concat((train, test))

df = pd.merge(df, users, how='left', on = 'user_id')

df.reset_index(drop=True, inplace=True)
df
df['open_flag'][73539:].value_counts()
import datetime

def convert_date(x):

    item = x.split(' ')[0]

    date = datetime.datetime.strptime(item, "%Y-%m-%d").date()

    num_days = (datetime.date(2020, 8, 1) - date).days

    return num_days



df['grass_date'] = df['grass_date'].apply(convert_date)

df['grass_date']
# ### Get variable from date

# df['grass_date'] = pd.to_datetime(df['grass_date'])



# df['day'] = df['grass_date'].dt.day.astype('category')

# df['dayofweek'] = df['grass_date'].dt.dayofweek.astype('category')

# df['month'] = df['grass_date'].dt.month.astype('category')



df.columns
col = ['login_count_last_10_days',

       'login_count_last_30_days', 'login_count_last_60_days',

       'checkout_count_last_10_days', 'checkout_count_last_30_days',

       'checkout_count_last_60_days']
df.boxplot(column=col, figsize=(20, 10))

#login_count_last_60_days

df['login_count_last_60_days'].values[df['login_count_last_60_days'].values > 900] = 900

df['login_count_last_30_days'].values[df['login_count_last_30_days'].values > 450] = 450

df['login_count_last_10_days'].values[df['login_count_last_10_days'].values > 200] = 200



#checkout_count_last_10_days

df['checkout_count_last_10_days'].values[df['checkout_count_last_10_days'].values > 150] = 150

df['checkout_count_last_30_days'].values[df['checkout_count_last_30_days'].values > 300] = 300

df['checkout_count_last_60_days'].values[df['checkout_count_last_60_days'].values > 650] = 650
#open_count_last_10_days_log

df['open_count_last_10_days_log'] = np.log(df['open_count_last_10_days'] + 1)

df['open_count_last_30_days_log'] = np.log(df['open_count_last_30_days'] + 1)

df['open_count_last_60_days_log'] = np.log(df['open_count_last_60_days'] + 1)
#age

### Make "age_class"

def make_age_class(dataset) :

    df = dataset.copy()

    

    # For NaN values

    df['age_class'] = df['age'].isna()

    df['age_class'] = df['age_class'].map({True:'Unknown',False:'<>'})

    

    # Make class for '>=30' and '<30' age

    df.loc[df['age']>=30, 'age_class'] = '>=30'

    df.loc[df['age']<30, 'age_class'] = '<30'

    

    return df



df = make_age_class(df)
# ### Make 'domain_type'

# list_low_domain = ['@163.com','@gmail.com','@yahoo.com','@ymail.com']

# list_med_domain = ['@outlook.com','@qq.com','@rocketmail.com']

# list_high_domain = ['@hotmail.com','@icloud.com','@live.com','other']



# def make_domain_type(dom) :

#     if dom in list_low_domain :

#         res = 'low_domain'

#     elif dom in list_med_domain :

#         res = 'med_domain'

#     elif dom in list_high_domain :

#         res = 'high_domain'

        

#     return res



# df['domain_type'] = df.apply(lambda x : make_domain_type(x['domain']), axis=1)
# df.drop(['row_id'], axis=1, inplace=True)
# df['domain'] = df['domain'].astype('category')


# import matplotlib.pyplot as plt



# plt.plot(df.index,df.row_id)

# plt.title('row_id - index')

# plt.xlabel('index')

# plt.ylabel('row_id')

# plt.show()
df.columns
df.replace(['Never checkout', 'Never open', 'Never login'], np.nan, inplace=True)
df.subject_line_length
df.columns
col = ['country_code', 'user_id', 'subject_line_length',

       'last_open_day', 'last_login_day', 'last_checkout_day',

       'open_count_last_10_days', 'open_count_last_30_days',

       'open_count_last_60_days', 'login_count_last_10_days',

       'login_count_last_30_days', 'login_count_last_60_days',

       'checkout_count_last_10_days', 'checkout_count_last_30_days',

       'checkout_count_last_60_days', 'open_flag', 'row_id', 'attr_1',

       'attr_2', 'attr_3', 'age', 'open_count_last_10_days_log',

       'open_count_last_30_days_log', 'open_count_last_60_days_log']
df[col] = df[col].astype('float')

df['domain'] = df['domain'].astype('category')

df['age_class'] = df['age_class'].astype('category')
### Initialize h2o

import h2o

h2o.init()
### Define predictor and response

X = ['country_code', 'grass_date', 'subject_line_length',

     'last_open_day', 'last_login_day', 'last_checkout_day',

     'open_count_last_10_days', 'open_count_last_30_days',

     'open_count_last_60_days', 'login_count_last_10_days',

     'login_count_last_30_days', 'login_count_last_60_days',

     'checkout_count_last_10_days', 'checkout_count_last_30_days',

     'checkout_count_last_60_days', 'attr_1', 'attr_2',

     'attr_3', 'age_class', 'open_count_last_10_days_log', 'domain',

     'open_count_last_30_days_log', 'open_count_last_60_days_log']

Y = 'open_flag'



list_col = X + [Y]
df_comb_train = df[:73539]

df_comb_test = df[73539:]
### Make H2O Frame

h2o_train = h2o.H2OFrame(df_comb_train[list_col])

h2o_test = h2o.H2OFrame(df_comb_test[X])
h2o_train[['age_class', 'domain']] = h2o_train[['age_class', 'domain']].asfactor()

h2o_test[['age_class', 'domain']] = h2o_test[['age_class', 'domain']].asfactor()

    

h2o_train[Y] = h2o_train[Y].asfactor()
### Make all H2O baseline model

from h2o.estimators.gbm import H2OGradientBoostingEstimator

from h2o.estimators.random_forest import H2ORandomForestEstimator

from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.estimators import H2OXGBoostEstimator

from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

import time



def h2o_compare_models(df_train, df_test, X, Y) :

    

    start = time.time()

    

    # Initialize all model (Ganti family/distributionnya)

    glm = H2OGeneralizedLinearEstimator(family='binomial', nfolds=5, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    gbm = H2OGradientBoostingEstimator(distribution='bernoulli', nfolds=5, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    xgb = H2OXGBoostEstimator(distribution='bernoulli', nfolds=5, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    lgbm = H2OXGBoostEstimator(distribution='bernoulli', tree_method="hist", grow_policy="lossguide",

                              nfolds=5, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    rf = H2ORandomForestEstimator(distribution='bernoulli', nfolds=5, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    ext = H2ORandomForestEstimator(distribution='bernoulli', histogram_type="Random",

                                  nfolds=5, keep_cross_validation_predictions=True, fold_assignment='Modulo')

    

    # Train model

    glm.train(x=X, y=Y, training_frame=df_train)

    gbm.train(x=X, y=Y, training_frame=df_train)

    xgb.train(x=X, y=Y, training_frame=df_train)

    lgbm.train(x=X, y=Y, training_frame=df_train)

    rf.train(x=X, y=Y, training_frame=df_train)

    ext.train(x=X, y=Y, training_frame=df_train)

    

    # Calculate train metrics (Bisa diganti)

    from sklearn.metrics import matthews_corrcoef

    train_glm = matthews_corrcoef(h2o_train[Y].as_data_frame(), glm.predict(h2o_train)['predict'].as_data_frame())

    train_gbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), gbm.predict(h2o_train)['predict'].as_data_frame())

    train_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.predict(h2o_train)['predict'].as_data_frame())

    train_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.predict(h2o_train)['predict'].as_data_frame())

    train_rf = matthews_corrcoef(h2o_train[Y].as_data_frame(), rf.predict(h2o_train)['predict'].as_data_frame())

    train_ext = matthews_corrcoef(h2o_train[Y].as_data_frame(), ext.predict(h2o_train)['predict'].as_data_frame())



    # Calculate CV metrics for all model (Bisa diganti)

    met_glm = matthews_corrcoef(h2o_train[Y].as_data_frame(), glm.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_gbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), gbm.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_rf = matthews_corrcoef(h2o_train[Y].as_data_frame(), rf.cross_validation_holdout_predictions()['predict'].as_data_frame())

    met_ext = matthews_corrcoef(h2o_train[Y].as_data_frame(), ext.cross_validation_holdout_predictions()['predict'].as_data_frame())

    

    # Make result dataframe

    result = pd.DataFrame({'Model':['GLM','GBM','XGB','LGBM','RF','ExtraTree'],

                          'Train Metrics':[train_glm,train_gbm,train_xgb,train_lgbm,train_rf,train_ext],

                          'CV Metrics':[met_glm,met_gbm,met_xgb,met_lgbm,met_rf,met_ext]})

    

    end = time.time()

    print('Time Used :',(end-start)/60)

    

    return result.sort_values('CV Metrics') 
### Compare models

res = h2o_compare_models(h2o_train, h2o_test, X, Y) 

res
### Make model for submission

lgbm = H2OXGBoostEstimator(distribution='bernoulli', tree_method="hist", grow_policy="lossguide",

                              nfolds=5, keep_cross_validation_predictions=True, fold_assignment='Modulo',

                              ntrees=1000, stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "logloss",

                              sample_rate=0.8, col_sample_rate=0.8, score_tree_interval=1)

lgbm.train(x=X, y=Y, training_frame=h2o_train)  
### Score

from sklearn.metrics import matthews_corrcoef

train_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.predict(h2o_train)['predict'].as_data_frame())

met_lgbm = matthews_corrcoef(h2o_train[Y].as_data_frame(), lgbm.cross_validation_holdout_predictions()['predict'].as_data_frame())



print('Train Score :',train_lgbm)

print('CV Score :',met_lgbm)
### Make submission

pred = lgbm.predict(h2o_test)['predict'].as_data_frame()

sub = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')

sub['open_flag'] = pred



sub.to_csv('subs_lgbm.csv', index=False)
sub
# xgb = xgb = H2OXGBoostEstimator(distribution='bernoulli', 

#                                 nfolds=5, 

#                                 keep_cross_validation_predictions=True, 

#                                 fold_assignment='Modulo')

# xgb.train(x=X, y=Y, training_frame=h2o_train)
# ### Score

# from sklearn.metrics import matthews_corrcoef

# train_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.predict(h2o_train)['predict'].as_data_frame())

# met_xgb = matthews_corrcoef(h2o_train[Y].as_data_frame(), xgb.cross_validation_holdout_predictions()['predict'].as_data_frame())



# print('Train Score :',train_xgb)

# print('CV Score :',met_xgb)
# ### Make submission

# pred = xgb.predict(h2o_test)['predict'].as_data_frame()

# sub = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')

# sub['open_flag'] = pred



# # sub.to_csv('subs_lgbm_2.csv', index=False)