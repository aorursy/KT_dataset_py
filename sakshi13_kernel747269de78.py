# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import subprocess

print('# Line count:')

for file in ['train.csv', 'test.csv', 'train_sample.csv']:

    lines = subprocess.run(['wc', '-l', '/kaggle/input/talkingdata-adtracking-fraud-detection/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')

    print(lines, end='', flush=True)
import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pal = sns.color_palette()
path = '/kaggle/input/talkingdata-adtracking-fraud-detection/'
dtypes = {

        'ip'            : 'uint32',

        'app'           : 'uint16',

        'device'        : 'uint16',

        'os'            : 'uint16',

        'channel'       : 'uint16',

        'is_attributed' : 'uint8',

        'click_id'      : 'uint32'

        }
train_sample_df = pd.read_csv(path+"train_sample.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], parse_dates=['click_time'])
train_sample_df.head()
train_sample_df.info()
train_sample_df.describe()
plt.figure(figsize=(15, 8))

cols = ['ip', 'app', 'device', 'os', 'channel']

uniques = [len(train_sample_df[col].unique()) for col in cols]

sns.set(font_scale=1.2)

ax = sns.barplot(cols, uniques, palette=pal, log=True)

ax.set(xlabel='Feature', ylabel='unique count', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center")
plt.figure(figsize=(8, 8))

sns.set(font_scale=1.2)

ax=sns.countplot(train_sample_df['is_attributed']);

ax.set(ylabel='Count of users', title='Count of users with App Downloaded vs Not Downloaded')
plt.figure(figsize=(6,6))

#sns.set(font_scale=1.2)

mean = (train_sample_df.is_attributed.values == 1).mean()

ax = sns.barplot(['App Downloaded (1)', 'Not Downloaded (0)'], [mean, 1-mean])

ax.set(ylabel='Percentage of users', title='Percentage of usres with App Downloaded vs Not Downloaded')

for p, uniq in zip(ax.patches, [mean, 1-mean]):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+0.01,

            '{}%'.format(round(uniq * 100, 2)),

            ha="center")
categorical = ['ip', 'app', 'device', 'os', 'channel']

for v in categorical:

    train_sample_df[v] = train_sample_df[v].astype('category')
#table to see ips with their associated total clicks

ip_repeat_df = train_sample_df['ip'].value_counts().reset_index(name='count_clicks')

ip_repeat_df.columns = ['ip', 'count_clicks']

ip_repeat_df[:10]
train_sample_df= train_sample_df.merge(ip_repeat_df, on='ip', how='left')

train_sample_df[train_sample_df['is_attributed']==1].sort_values('count_clicks', ascending=False)[:10]
train_sample_df[train_sample_df['is_attributed']==1].ip.describe()
train_sample_df['click_hour']=train_sample_df['click_time'].dt.hour

train_sample_df.head()
train_sample_df[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).count().plot()

plt.title('HOURLY CLICK FREQUENCY Lineplot');

plt.ylabel('Number of Clicks');
train_sample_df[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).mean().plot()

plt.title('HOURLY CLICK FREQUENCY Lineplot');

plt.ylabel('Number of Clicks');
train_sample_df['click_DAY']=train_sample_df['click_time'].dt.day

train_sample_df[['click_DAY','is_attributed']].groupby(['click_DAY'], as_index=True).count().plot(kind='bar', color='blue')

plt.title('Daily CLICK FREQUENCY BARPLOT');

plt.ylabel('Number of Clicks')
train_df = pd.read_csv(path+"train.csv",  nrows=30000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
test_df = pd.read_csv(path+"test.csv", dtype=dtypes,skiprows=range(1,11290470), nrows=7500000, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
print(train_df.head())

print(test_df.head())
variables = ['ip', 'app', 'device', 'os', 'channel']

for v in variables:

    train_df[v] = train_df[v].astype('category')

    test_df[v]=test_df[v].astype('category')
train_df.info()
train_df['click_time'] = pd.to_datetime(train_df['click_time'])

test_df['click_time'] = pd.to_datetime(test_df['click_time'])
train_df['click_hour']=train_df['click_time'].dt.hour

test_df['click_hour']=test_df['click_time'].dt.hour
#check for hourly patterns of training data

train_df[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).count().plot(kind='bar', color='blue')

plt.title('HOURLY CLICK FREQUENCY BARPLOT');

plt.ylabel('Number of Clicks')



train_df[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).mean().plot(kind='bar', color='green')

plt.title('HOURLY CLICK FREQUENCY BARPLOT');

plt.ylabel('Number of Clicks')



train_df[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).count().plot(color='blue')

plt.title('HOURLY CLICK FREQUENCY LINEPLOT');

plt.ylabel('Number of Clicks');



train_df[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).mean().plot(color='green')

plt.title('HOURLY CONVERSION RATIO LINEPLOT');

plt.ylabel('Converted Ratio');
train_df['click_DAY']=train_df['click_time'].dt.day
train_df[['click_DAY','is_attributed']].groupby(['click_DAY'], as_index=True).count().plot(kind='bar', color='blue')

plt.title('Daily CLICK FREQUENCY BARPLOT');

plt.ylabel('Number of Clicks')
train_samples = len(train_df)

train_df=train_df.append(test_df)
import gc
del test_df

gc.collect()
temp_df = train_df[['ip','click_hour','channel']].groupby(by=['ip','click_hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_hour_count'})

train_df = train_df.merge(temp_df, on=['ip','click_hour'], how='left')

del temp_df

gc.collect()

train_df.head()
temp_df = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})

train_df = train_df.merge(temp_df, on=['ip','app'], how='left')

del temp_df

gc.collect()

train_df.head()
temp_df = train_df[['ip', 'app','os', 'channel']].groupby(by=['ip', 'app','os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})

train_df = train_df.merge(temp_df, on=['ip','app','os'], how='left')

del temp_df

gc.collect()

train_df.head()
temp_df = train_df[['ip', 'device','os', 'channel']].groupby(by=['ip', 'device','os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_device_os_count'})

train_df = train_df.merge(temp_df, on=['ip','device','os'], how='left')

del temp_df

gc.collect()

train_df.head()
import lightgbm as lgb



def lgb_modelfit(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',

                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):

    lgb_params = {

        'boosting_type': 'gbdt',

        'objective': objective,

        'metric':metrics,

        'learning_rate': 0.01,

        'num_leaves': 31,  

        'max_depth': -1,  # -1 means no limit

        'min_child_samples': 20,  

        'max_bin': 255,  

        'subsample': 0.6,  

        'subsample_freq': 0, 

        'colsample_bytree': 0.3,  

        'min_child_weight': 5,  

        'subsample_for_bin': 200000,  

        'min_split_gain': 0,  

        'reg_alpha': 0,  

        'reg_lambda': 0,  

        'nthread': 4,

        'verbose': 0,

        'metric':metrics

    }



    lgb_params.update(params)



    print("preparing validation datasets")



    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,

                          feature_name=predictors,

                          categorical_feature=categorical_features

                          )

    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,

                          feature_name=predictors,

                          categorical_feature=categorical_features

                          )



    evals_results = {}



    bst1 = lgb.train(lgb_params, 

                     xgtrain, 

                     valid_sets=[xgtrain, xgvalid], 

                     valid_names=['train','valid'], 

                     evals_result=evals_results, 

                     num_boost_round=num_boost_round,

                     early_stopping_rounds=early_stopping_rounds,

                     verbose_eval=10, 

                     feval=feval)



    n_estimators = bst1.best_iteration

    print("\nModel Report")

    print("n_estimators : ", n_estimators)

    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])



    return bst1
train_df.info()
test_df = train_df[train_samples:]

val_df = train_df[(train_samples-3000000):train_samples]

train_df = train_df[:(train_samples-3000000)]



print("train size: ", len(train_df))

print("valid size: ", len(val_df))

print("test size : ", len(test_df))
# removed IP from features as the ip has been mainly used to extract new features. 

target = 'is_attributed'

predictors = ['app','device','os', 'channel', 'click_hour',  

              'ip_hour_count', 'ip_app_count', 'ip_app_os_count','ip_device_os_count' ]

categorical = ['app', 'device', 'os', 'channel', 'click_hour']



test_reference = pd.DataFrame()

test_reference['click_id'] = test_df['click_id'].astype('int')

import time

print("Training...")

start_time = time.time()





params = {

    'learning_rate': 0.15,

    'num_leaves': 7,  

    'max_depth': 3, 

    'min_child_samples': 100,  

    'max_bin': 100, 

    'subsample': 0.7,  

    'subsample_freq': 1,  

    'colsample_bytree': 0.9, 

    'min_child_weight': 0,  

    'scale_pos_weight':99 # because training data is extremely unbalanced 

}

bst = lgb_modelfit(params, 

                        train_df, 

                        val_df, 

                        predictors, 

                        target, 

                        objective='binary', 

                        metrics='auc',

                        early_stopping_rounds=30, 

                        verbose_eval=True, 

                        num_boost_round=500, 

                        categorical_features=categorical)



print('[{}]: model training time'.format(time.time() - start_time))
print("Calculating predictions")

test_reference['is_attributed'] = bst.predict(test_df[predictors])

print("writing the results to test_predictions.csv")

test_reference.to_csv('test_predictions.csv',index=False)

print("Predictions calculated and written into csv file")
test_reference.tail()
plt.figure(figsize=(6,6))

#sns.set(font_scale=1.2)

mean = (test_reference.is_attributed.values == 1).mean()

ax = sns.barplot(['App Downloaded (1)', 'Not Downloaded (0)'], [mean, 1-mean])

ax.set(ylabel='Percentage of users', title='Percentage of users with App Downloaded vs Not Downloaded')

for p, uniq in zip(ax.patches, [mean, 1-mean]):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+0.01,

            '{}%'.format(round(uniq * 100, 2)),

            ha="center")
lgb.plot_importance(bst, importance_type='split')

test_df.head()