# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import gc



import datetime



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(pd.read_csv('../input/train.csv'))

test = reduce_mem_usage(pd.read_csv('../input/test.csv'))

train.head()
train.describe()
test.describe()
nm_df = reduce_mem_usage(pd.read_csv('../input/new_merchant_transactions.csv'))

print(nm_df.shape)

nm_df.head()
nm_df['purchase_date'] = pd.to_datetime(nm_df.purchase_date)



nm_df['purch_year'] = nm_df.purchase_date.dt.year

nm_df['purch_mon'] = nm_df.purchase_date.dt.month

nm_df['purch_dow'] = nm_df.purchase_date.dt.dayofweek

nm_df['purch_wk'] = nm_df.purchase_date.dt.week

nm_df['purch_day'] = nm_df.purchase_date.dt.dayofyear



## adding in a few more features - 2.6.19

## source: https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

nm_df['purch_woy'] = nm_df.purchase_date.dt.weekofyear

nm_df['purch_wknd'] = (nm_df.purchase_date.dt.weekday >=5).astype(int)

nm_df['purch_hr'] = nm_df.purchase_date.dt.hour

#https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

nm_df['purch_month_diff'] = ((datetime.datetime.today() - nm_df.purchase_date).dt.days)//30

nm_df['purch_month_diff'] += nm_df['month_lag']
nm_df.head()
m_df = reduce_mem_usage(pd.read_csv('../input/merchants.csv'))

print(m_df.shape)

m_df.head()
nmm_df = nm_df.merge(m_df, on='merchant_id', how='outer')

nmm_df.shape
for c in nmm_df.columns:

    if nmm_df[c].dtype == 'object':

        vcs = nmm_df[c].value_counts()

        if len(vcs) < 20: ## checking if theres a lot of unique values

            print(vcs)

#     print(c,'->',nmm_df[c].dtype)
nmm_df['authorized_flag'] = (nmm_df.authorized_flag == 'Y').astype(int)

nmm_df['category_1_x'] = (nmm_df.category_1_x == 'Y').astype(int)

nmm_df['purchase_date'] = pd.to_datetime(nmm_df.purchase_date)

nmm_df['category_1_y'] = (nmm_df.category_1_y == 'Y').astype(int)

nmm_df['category_4'] = (nmm_df.category_4 == 'Y').astype(int)
mrpr = pd.get_dummies(nmm_df.most_recent_purchases_range)

mrsr = pd.get_dummies(nmm_df.most_recent_sales_range)

cat3 = pd.get_dummies(nmm_df.category_3)
for d in [mrpr, mrsr, cat3]:

    for c in d.columns:

        nmm_df[c] = d[c]

        

nmm_df.drop(['most_recent_purchases_range','most_recent_sales_range','category_3'], axis=1, inplace=True)



try:

    del mrpr, mrsr, cat3, nm_df, m_df

except:

    pass

gc.collect()
nmm_df.isnull().sum()/len(nmm_df)
## fill NA with most common value

for c in nmm_df.columns:

    if nmm_df[c].isnull().sum()/len(nmm_df[c]) > 0:

        nmm_df[c] = nmm_df[c].fillna(nmm_df[c].value_counts().index[0])

        

# nmm_df.isnull().sum()/len(nmm_df)
## factorize merchant id



nmm_df['merchant_id'] = pd.factorize(nmm_df['merchant_id'])[0]
nmm_agg = nmm_df.groupby('card_id').agg(['sum','mean','max','min','var',

                                        'median','count','skew','nunique',#'mode'

                                       ])

print(nmm_agg.shape)



try:

    del nmm_df

except:

    pass

gc.collect()
hist = reduce_mem_usage(pd.read_csv('../input/historical_transactions.csv'))

hist.head()
hist['authorized_flag'] = (hist.authorized_flag == 'Y').astype(int)

hist['category_1'] = (hist.category_1 == 'Y').astype(int)
cat3 = pd.get_dummies(hist.category_3)



for c in cat3.columns:

    hist[c] = cat3[c]

hist.drop('category_3',axis=1, inplace=True)



del cat3
hist['purchase_date'] = pd.to_datetime(hist.purchase_date)



hist['histpurch_year'] = hist.purchase_date.dt.year

hist['histpurch_mon'] = hist.purchase_date.dt.month

hist['histpurch_dow'] = hist.purchase_date.dt.dayofweek

hist['histpurch_wk'] = hist.purchase_date.dt.week

hist['histpurch_day'] = hist.purchase_date.dt.dayofyear



## adding in a few more features - 2.6.19

## source: https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

hist['histpurch_woy'] = hist.purchase_date.dt.weekofyear

hist['histpurch_wknd'] = (hist.purchase_date.dt.weekday >=5).astype(int)

hist['histpurch_hr'] = hist.purchase_date.dt.hour

#https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

hist['histpurch_month_diff'] = ((datetime.datetime.today() - hist.purchase_date).dt.days)//30

hist['histpurch_month_diff'] += hist['month_lag']
print(hist.shape)
hist.isnull().sum()/len(hist)
## fill NA with most common value

for c in hist.columns:

    if hist[c].isnull().sum()/len(hist[c]) > 0:

        hist[c] = hist[c].fillna(hist[c].value_counts().index[0])

        



#factorize merchant id

hist['merchant_id'] = pd.factorize(hist['merchant_id'])[0]
hist_agg = hist.groupby('card_id').agg(['sum','mean','max','min','var',

                                        'median','count','skew','nunique',#'mode'

                                       ])

print(hist_agg.shape)
try:

    del hist

except:

    pass

gc.collect()
train_agg = train.set_index('card_id').join(hist_agg, how='left').join(nmm_agg, how='left', rsuffix='_nmm').fillna(0)

print(train_agg.shape)
test_agg = test.set_index('card_id').join(hist_agg, how='left').join(nmm_agg, how='left', rsuffix='_nmm').fillna(0)

print(test_agg.shape)
train_agg.columns = [''.join(col).strip() for col in train_agg.columns.values]

test_agg.columns = [''.join(col).strip() for col in test_agg.columns.values]

train_agg.columns = train_agg.columns.str.replace(' ','')

test_agg.columns = test_agg.columns.str.replace(' ','')

train_agg.head(3)
# train_agg = train_agg.reset_index()

try:

    del train, test

except:

    pass



gc.collect()

train_agg['month'] = train_agg['first_active_month'].str[-2:]

train_agg['year'] = train_agg['first_active_month'].str[:-3]

train_agg.drop(['first_active_month'],axis=1,inplace=True)

train_agg['month'] = train_agg.month.astype(int)

train_agg['year'] = train_agg.year.astype(int)
test_agg['month'] = test_agg['first_active_month'].str[-2:]

test_agg['year'] = test_agg['first_active_month'].str[:-3]

test_agg.drop(['first_active_month'],axis=1,inplace=True)

test_agg['month'] = test_agg.month.fillna(0).astype(int)

test_agg['year'] = test_agg.year.fillna(0).astype(int)
for c in train_agg.columns:

    if np.isinf(train_agg[c]).sum()/len(train_agg[c]) > 0:

        print(c)

        train_agg[c] = train_agg[c].replace([np.inf, -np.inf], train_agg[c].value_counts().index[0])
for c in test_agg.columns:

    if np.isinf(test_agg[c]).sum()/len(test_agg[c]) > 0:

        print(c)

        test_agg[c] = test_agg[c].replace([np.inf, -np.inf], test_agg[c].value_counts().index[0])
for c in train_agg.columns:

    if train_agg[c].isnull().sum()/len(train_agg[c]) > 0:

        print(c)

        train_agg[c] = train_agg[c].fillna(train_agg[c].value_counts().index[0])
for c in test_agg.columns:

    if test_agg[c].isnull().sum()/len(test_agg[c]) > 0:

        print(c)

        test_agg[c] = test_agg[c].fillna(test_agg[c].value_counts().index[0])
from sklearn.model_selection import StratifiedKFold, RepeatedKFold

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



from sklearn.linear_model import HuberRegressor, LassoLars, RANSACRegressor

from sklearn.ensemble import GradientBoostingRegressor
## trying out various methods to grrab the outliers

## The Huber Regressor errored out. Lars didnt improve anything nor did RANSAC.

## Also trying a gradient boosting decision stump with huber loss

##   GBT stumps tend to fit to outliers more as well

##      On v18, we will do a little bigger than a stump w maxdepth of 2



llars = LassoLars(alpha=0.9, fit_intercept=True, normalize=True, 

                  precompute="auto", max_iter=500)



rr = RANSACRegressor(#llars,

                     random_state=123)



gbr = GradientBoostingRegressor(loss='huber',

                                learning_rate=0.01, #0.02,

                                n_estimators=1000, #500,

                                subsample=0.8, 

                                max_depth=2, #1, 

                                random_state=123, max_features=None, alpha=0.9,

                                validation_fraction=0.1)



gc.collect()
# del x0, y, trn_data, val_data

gc.collect()
folds = StratifiedKFold(n_splits = 5, shuffle = True)

train_predictions = np.zeros(len(train_agg))

test_predictions = np.zeros(len(test_agg))

lin_preds = np.zeros(len(test_agg))

n_fold = 0

for train_index, test_index in folds.split(train_agg, train_agg['feature_1']):

    ## from https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending

    n_fold += 1

    param ={

        'task': 'train',

        'boosting': 'goss',

        'objective': 'regression',

        'metric': 'rmse',

        'learning_rate': 0.01,

        'subsample': 0.9855232997390695,

        'max_depth': 7,

        'top_rate': 0.9064148448434349,

        'num_leaves': 63,

        'min_child_weight': 41.9612869171337,

        'other_rate': 0.0721768246018207,

        'reg_alpha': 9.677537745007898,

        'colsample_bytree': 0.5665320670155495,

        'min_split_gain': 9.820197773625843,

        'reg_lambda': 8.2532317400459,

        'min_data_in_leaf': 21,

        'verbose': -1,

        'seed':int(2**n_fold),

        'bagging_seed':int(2**n_fold),

        'drop_seed':int(2**n_fold)

        }

    

    y = train_agg['target']

    x0 = train_agg.drop('target', axis = 1)

    

    trn_data = lgb.Dataset(x0.iloc[train_index], label=y.iloc[train_index])

    val_data = lgb.Dataset(x0.iloc[test_index], label=y.iloc[test_index])

    

    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = 200)

    train_predictions[test_index] = clf.predict(x0.iloc[test_index], num_iteration=clf.best_iteration)



    test_predictions += clf.predict(test_agg, num_iteration=clf.best_iteration) / folds.n_splits

    

#     llars.fit(x0.iloc[train_index], y.iloc[train_index])

#     rr.fit(x0.iloc[train_index], y.iloc[train_index])

    gbr.fit(x0.iloc[train_index], y.iloc[train_index]) ## Decision Stump on Huber to Grab Outliers?

    lin_preds += gbr.predict(test_agg)/folds.n_splits



print("LGB Train Error:",np.sqrt(mean_squared_error(train_predictions, y)))

gc.collect()
pd.concat([pd.DataFrame(test_predictions[:10], columns=['lgb']),

           pd.DataFrame(lin_preds[:10], columns=['lin'])], axis=1, join='outer')



# print(test_predictions[:10])

# print('/n')

# print(lin_preds[:10])
combined_preds = 0.9*test_predictions + 0.1*lin_preds

combined_preds
predictions = pd.DataFrame(

    data = {

        'card_id' : test_agg.index, #test_agg['card_id'],

        'target' : combined_preds, #test_predictions

    }

)

predictions.to_csv('submit.csv', index = False) 
predictions.head(10)
len(predictions)
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,1,figsize=(14,12))

lgb.plot_importance(clf, max_num_features=50, ax=ax);