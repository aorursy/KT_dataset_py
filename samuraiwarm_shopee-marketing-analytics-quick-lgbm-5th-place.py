import numpy as np

import pandas as pd

import math

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import lightgbm as gbm

import time
users = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/users.csv')

enc = LabelEncoder()

enc.fit(users['domain'])

users['domain'] = enc.transform(users['domain'])

train = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/train.csv')

df = train.join(users, on='user_id', rsuffix='_right').drop(columns=['user_id','user_id_right','row_id'])



# replace with dummy very high values of days

df = df.replace(['Never open','Never login','Never checkout'],1000000)

# some people have negative age for some reason(?)

df['age'] = np.absolute(df['age'])

# conservatively minimum legal age

# and removing people that fill in the maximum age possible (118 years old)

df.loc[(df['age']<16) | (df['age']>80),'age'] = np.nan

df[['last_open_day','last_login_day','last_checkout_day']] = df[['last_open_day','last_login_day','last_checkout_day']].astype(int)

df['grass_date'] = pd.to_datetime(df['grass_date']).astype(int)



# generating features

# first difference on 30-10 days, and 60-30 days

df['open_count_change_30_10'] = df['open_count_last_30_days']-df['open_count_last_10_days']

df['login_count_change_30_10'] = df['login_count_last_30_days']-df['login_count_last_10_days']

df['checkout_count_change_30_10'] = df['checkout_count_last_30_days']-df['checkout_count_last_10_days']

df['open_count_change_60_30'] = df['open_count_last_60_days']-df['open_count_last_30_days']

df['login_count_change_60_30'] = df['login_count_last_60_days']-df['login_count_last_30_days']

df['checkout_count_change_60_30'] = df['checkout_count_last_60_days']-df['checkout_count_last_30_days']



# rate of user activities

df['open_rate_last_10'] = df['open_count_last_10_days']/10

df['login_rate_last_10'] = df['login_count_last_10_days']/10

df['checkout_rate_last_10'] = df['checkout_count_last_10_days']/10

df['open_rate_last_30'] = df['open_count_last_30_days']/30

df['login_rate_last_30'] = df['login_count_last_30_days']/30

df['checkout_rate_last_30'] = df['checkout_count_last_30_days']/30

df['open_rate_last_60'] = df['open_count_last_60_days']/60

df['login_rate_last_60'] = df['login_count_last_60_days']/60

df['checkout_rate_last_60'] = df['checkout_count_last_60_days']/60



# first diff

df['open_rate_change_30_10'] = df['open_rate_last_30']-df['open_rate_last_10']

df['login_rate_change_30_10'] = df['login_rate_last_30']-df['login_rate_last_10']

df['checkout_rate_change_30_10'] = df['checkout_rate_last_30']-df['checkout_rate_last_10']

df['open_rate_change_60_30'] = df['open_rate_last_60']-df['open_rate_last_30']

df['login_rate_change_60_30'] = df['login_rate_last_60']-df['login_rate_last_30']

df['checkout_rate_change_60_30'] = df['checkout_rate_last_60']-df['checkout_rate_last_30']



# second diff

df['open_2rate_change'] = df['open_rate_change_60_30']/30-df['open_rate_change_30_10']/20

df['login_2rate_change'] = df['login_rate_change_60_30']/30-df['login_rate_change_30_10']/20

df['checkout_2rate_change'] = df['checkout_rate_change_60_30']/30-df['checkout_rate_change_30_10']/20



# all these have linear relation with open/login/checkout count, ignore these

df = df.drop(columns=[

    'open_rate_last_10',

    'login_rate_last_10',

    'checkout_rate_last_10',

    'open_rate_last_30',

    'login_rate_last_30',

    'checkout_rate_last_30',

    'open_rate_last_60',

    'login_rate_last_60',

    'checkout_rate_last_60',

])

df
X = df.drop(columns='open_flag')

y = df['open_flag']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
params = {'task':'train',

          'boosting_type':'gbdt',

          'num_leaves':5,

          'max_depth':5,

          'learning_rate':0.1,

          'n_estimators':500,

          'subsample_for_bin':200000,

          'objective':'binary',

#           'scale_pos_weight':(1+62083/11456),

          'min_split_gain':0.003,

          'min_child_weight':0.001,

          'min_child_samples':10,

          'subsample':0.8,

          'subsample_freq':50,

          'reg_alpha':30,

          'reg_lambda':30,

          'feature_fraction':0.8,

          'is_unbalance':True,

          'boost_from_average':False,

         }

X_train_ds = gbm.Dataset(X_train, label=y_train)



start = time.time()

est = gbm.train(params, X_train_ds)

print(time.time()-start)

y_train_pred_cont = est.predict(X_train)

y_test_pred_cont = est.predict(X_test)

y_train_pred = (y_train_pred_cont>0.75).astype(int)

y_test_pred = (y_test_pred_cont>0.75).astype(int)

tn = ((y_train==0) & (y_train_pred==0)).sum()

fp = ((y_train==0) & (y_train_pred==1)).sum()

fn = ((y_train==1) & (y_train_pred==0)).sum()

tp = ((y_train==1) & (y_train_pred==1)).sum()

print(tn,fp,fn,tp,((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))

tn = ((y_test==0) & (y_test_pred==0)).sum()

fp = ((y_test==0) & (y_test_pred==1)).sum()

fn = ((y_test==1) & (y_test_pred==0)).sum()

tp = ((y_test==1) & (y_test_pred==1)).sum()

print(tn,fp,fn,tp,((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))

print((y_train==y_train_pred).sum()/y_train.count())

print((y_test==y_test_pred).sum()/y_test.count())
X_ds = gbm.Dataset(X, label=y)

start = time.time()

est = gbm.train(params, X_ds)

print(time.time()-start)

y_train_pred_cont = est.predict(X)

y_train_pred = (y_train_pred_cont>0.741).astype(int)

tn = ((y==0) & (y_train_pred==0)).sum()

fp = ((y==0) & (y_train_pred==1)).sum()

fn = ((y==1) & (y_train_pred==0)).sum()

tp = ((y==1) & (y_train_pred==1)).sum()

print(tn,fp,fn,tp,((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
for thresh in np.linspace(0,1,101):

    y_train_pred = (y_train_pred_cont>thresh).astype(int)

    tn = ((y==0) & (y_train_pred==0)).sum()

    fp = ((y==0) & (y_train_pred==1)).sum()

    fn = ((y==1) & (y_train_pred==0)).sum()

    tp = ((y==1) & (y_train_pred==1)).sum()

    print(thresh,tn,fp,fn,tp,((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
for thresh in np.linspace(0.74,0.76,101):

    y_train_pred = (y_train_pred_cont>thresh).astype(int)

    tn = ((y==0) & (y_train_pred==0)).sum()

    fp = ((y==0) & (y_train_pred==1)).sum()

    fn = ((y==1) & (y_train_pred==0)).sum()

    tp = ((y==1) & (y_train_pred==1)).sum()

    print(thresh,tn,fp,fn,tp,((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
test = pd.read_csv('/kaggle/input/student-shopee-code-league-marketing-analytics/test.csv')

df_test = test.join(users, on='user_id', rsuffix='_right').drop(columns=['user_id','user_id_right','row_id'])

df_test = df_test.replace(['Never open','Never login','Never checkout'],1000000)

df_test['age'] = np.absolute(df_test['age'])

df_test.loc[(df_test['age']<16) | (df_test['age']>80),'age'] = np.nan

df_test[['last_open_day','last_login_day','last_checkout_day']] = df_test[['last_open_day','last_login_day','last_checkout_day']].astype(int)

df_test['grass_date'] = pd.to_datetime(df_test['grass_date']).astype(int)

df_test['open_count_change_30_10'] = df_test['open_count_last_30_days']-df_test['open_count_last_10_days']

df_test['login_count_change_30_10'] = df_test['login_count_last_30_days']-df_test['login_count_last_10_days']

df_test['checkout_count_change_30_10'] = df_test['checkout_count_last_30_days']-df_test['checkout_count_last_10_days']

df_test['open_count_change_60_30'] = df_test['open_count_last_60_days']-df_test['open_count_last_30_days']

df_test['login_count_change_60_30'] = df_test['login_count_last_60_days']-df_test['login_count_last_30_days']

df_test['checkout_count_change_60_30'] = df_test['checkout_count_last_60_days']-df_test['checkout_count_last_30_days']

df_test['open_rate_last_10'] = df_test['open_count_last_10_days']/10

df_test['login_rate_last_10'] = df_test['login_count_last_10_days']/10

df_test['checkout_rate_last_10'] = df_test['checkout_count_last_10_days']/10

df_test['open_rate_last_30'] = df_test['open_count_last_30_days']/30

df_test['login_rate_last_30'] = df_test['login_count_last_30_days']/30

df_test['checkout_rate_last_30'] = df_test['checkout_count_last_30_days']/30

df_test['open_rate_last_60'] = df_test['open_count_last_60_days']/60

df_test['login_rate_last_60'] = df_test['login_count_last_60_days']/60

df_test['checkout_rate_last_60'] = df_test['checkout_count_last_60_days']/60

df_test['open_rate_change_30_10'] = df_test['open_rate_last_30']-df_test['open_rate_last_10']

df_test['login_rate_change_30_10'] = df_test['login_rate_last_30']-df_test['login_rate_last_10']

df_test['checkout_rate_change_30_10'] = df_test['checkout_rate_last_30']-df_test['checkout_rate_last_10']

df_test['open_rate_change_60_30'] = df_test['open_rate_last_60']-df_test['open_rate_last_30']

df_test['login_rate_change_60_30'] = df_test['login_rate_last_60']-df_test['login_rate_last_30']

df_test['checkout_rate_change_60_30'] = df_test['checkout_rate_last_60']-df_test['checkout_rate_last_30']

df_test['open_2rate_change'] = df_test['open_rate_change_60_30']/30-df_test['open_rate_change_30_10']/20

df_test['login_2rate_change'] = df_test['login_rate_change_60_30']/30-df_test['login_rate_change_30_10']/20

df_test['checkout_2rate_change'] = df_test['checkout_rate_change_60_30']/30-df_test['checkout_rate_change_30_10']/20

df_test = df_test.drop(columns=[

    'open_rate_last_10',

    'login_rate_last_10',

    'checkout_rate_last_10',

    'open_rate_last_30',

    'login_rate_last_30',

    'checkout_rate_last_30',

    'open_rate_last_60',

    'login_rate_last_60',

    'checkout_rate_last_60',

])

df_test
threshold = 0.7554

y_test_pred_cont = est.predict(df_test)

y_test_pred = (y_test_pred_cont>threshold).astype(int)

y_test_pred
y_test_pred.sum()
pd.DataFrame(

    {'row_id':[i for i in range(0,55970)],'open_flag':y_test_pred}

).to_csv('/kaggle/working/submission.csv',index=False)
# feature importances

sorted(list(zip(est.feature_name(),est.feature_importance(importance_type='gain'))),key=lambda x:x[1])[::-1]