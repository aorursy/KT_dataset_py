# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.simplefilter(action='ignore')

import gc

import os

import time



import plotly

import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected = True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print("{} observations and {} features in train set.".format(train_df.shape[0],train_df.shape[1]))

print("{} observations and {} features in test set.".format(test_df.shape[0],test_df.shape[1]))
train_df.head(5)
test_df = pd.read_csv('../input/test.csv')

test_df.head(5)
active_month_series_traindf = train_df['first_active_month'].value_counts()

active_month_series_testdf = test_df['first_active_month'].value_counts()



trace0 = go.Scatter(

        x = active_month_series_traindf.index,

        y = active_month_series_traindf.values,

        name = 'train set')



trace1 = go.Scatter(

        x = active_month_series_testdf.index,

        y = active_month_series_testdf.values,

        name = 'test set')



plotly.offline.iplot({

    "data": [trace0, trace1],

    "layout": go.Layout(title="First Active month in Train & Test data")

})
train_df['target'] = pd.to_numeric(train_df['target'], errors = 'ignore')

train_df['target'].dtypes
print("maximum loyalt score is {}.".format(train_df['target'].max()))

print("minimum loyalt score is {}.".format(train_df['target'].min()))

print("mean loyalt score is {}.".format(train_df['target'].mean()))

print("std loyalt score is {}.".format(train_df['target'].std()))
train_df['target'].describe()
train_df.describe()
target_bins = pd.cut(train_df['target'],[-35,-30,-20,-10,-5,0,5,10,20,30])

Loyalt_score = train_df.groupby(target_bins)['target'].agg(['count']).reset_index()





fig, ax = plt.subplots(1,2,figsize=(15,5))

ax[0].set_title('Bar plot of Loyalt Score distribution')

vis1 = sns.countplot(x=target_bins, data=Loyalt_score, ax = ax[0])





ax[1].set_title('Histogram of Loyalt Score distribution')

vis2 = sns.distplot(train_df['target'].values, bins=50, kde=False, color="red", ax = ax[1])
fig, ax = plt.subplots(1,3, figsize=(20,5))

vis1 = sns.countplot(x = 'feature_1', data = train_df, ax = ax[0])

vis2 = sns.countplot(x = 'feature_2', data = train_df, ax = ax[1])

vis2 = sns.countplot(x = 'feature_3', data = train_df, ax = ax[2])

ax[0].set_title('Feature_1 in train set')

ax[1].set_title('Feature_2 in train set')

ax[2].set_title('Feature_3 in train set')
from plotly import tools 



feature1_traindf = train_df['feature_1'].value_counts()

feature1_testdf = test_df['feature_1'].value_counts()

feature2_traindf = train_df['feature_2'].value_counts()

feature2_testdf = test_df['feature_2'].value_counts()

feature3_traindf = train_df['feature_3'].value_counts()

feature3_testdf = test_df['feature_3'].value_counts()





fig = tools.make_subplots(rows=1, cols=3)



trace0 = go.Bar(x = feature1_traindf.index, y = feature1_traindf.values, name = 'feature1 train set')

trace1 = go.Bar(x = feature1_testdf.index, y = feature1_testdf.values, name = 'feature1 test set')

trace2 = go.Bar(x = feature2_traindf.index, y = feature2_traindf.values, name = 'feature2 train set')

trace3 = go.Bar(x = feature2_testdf.index, y = feature2_testdf.values, name = 'feature2 test set')

trace4 = go.Bar(x = feature3_traindf.index, y = feature3_traindf.values, name = 'feature3 train set')

trace5 = go.Bar(x = feature3_testdf.index, y = feature3_testdf.values, name = 'feature3 test set')





fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 3)

fig.append_trace(trace5, 1, 3)





fig['layout'].update(height=400, width=1100, title='Feature 1, 2 3 in train and test set')

plotly.offline.iplot(fig, filename='stacked-subplots')
merchants_df = pd.read_csv('../input/merchants.csv')

print('merchant datasets')

merchants_df.head(5)
print("{} observations and {} features in merchants data.".format(merchants_df.shape[0],merchants_df.shape[1]))
sale_range = merchants_df['most_recent_sales_range'].value_counts()

purchanse_range = merchants_df['most_recent_purchases_range'].value_counts()



plotly.offline.iplot({

    "data": [go.Scatter(x = sale_range.index, y = sale_range.values, name = 'sales range'),

            go.Scatter(x = purchanse_range.index, y = purchanse_range.values, name = 'purchases range')],

    "layout": go.Layout(title="most_recent_sales_purchases_range")

})
lag3 = merchants_df['active_months_lag3'].value_counts()

lag6 = merchants_df['active_months_lag6'].value_counts()

lag12 = merchants_df['active_months_lag12'].value_counts()



plotly.offline.iplot({

    "data": [go.Scatter(x = lag3.index, y = lag3.values, name = 'lag3'),

            go.Scatter(x = lag6.index, y = lag6.values, name = 'lag6'),

            go.Scatter(x = lag12.index, y = lag12.values, name = 'lag12')],

    "layout": go.Layout(title="active_months_lag3, lag6, lag12 data")

})
hist_df = pd.read_csv('../input/historical_transactions.csv')

hist_df.head(5)
print("{} observations and {} features in history merchant data.".format(hist_df.shape[0],hist_df.shape[1]))
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if df[col].dtypes == 'object':

            df[col] = df[col].astype('category')

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
new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")

new_trans_df.head()
historical_transactions = reduce_mem_usage(hist_df)

newMerchant_transactions = reduce_mem_usage(new_trans_df)

gc.collect()
historical_transactions.dtypes
histPurchase_amount = pd.DataFrame({'hist_total_transactions' : historical_transactions.groupby(['card_id','authorized_flag'])['purchase_amount'].size()}).reset_index()

histPurchase_amount.head(5)
authorised_count = pd.DataFrame({'hist_authorised_count' : histPurchase_amount.groupby('card_id')['authorized_flag'].count()}).reset_index()

histPurchase_amount = pd.merge(authorised_count, histPurchase_amount, on = 'card_id', how = 'left')

histPurchase_amount.head(5)
newPurchase_amount = pd.DataFrame({'new_total_transactions' : newMerchant_transactions.groupby(['card_id','authorized_flag'])['purchase_amount'].size()}).reset_index()

newPurchase_amount.head(5)
newAuthorised_count = pd.DataFrame({'new_authorised_count' : newPurchase_amount.groupby('card_id')['authorized_flag'].count()}).reset_index()

newPurchase_amount = pd.merge(newAuthorised_count, newPurchase_amount, on = 'card_id', how = 'left')

newPurchase_amount.head(5)
print("Unauthorised entries in new Transactions are: {}\nOnly Unauthorised entries in history Transactions are: {}"

      .format(newPurchase_amount.loc[newPurchase_amount['authorized_flag'] == 'N'].shape[0],

               histPurchase_amount.loc[((histPurchase_amount['authorized_flag'] == 'N') & (histPurchase_amount['hist_authorised_count'] != 2))].shape[0]))
# All card holders has atleast once authorised entires so.

del histPurchase_amount

del newPurchase_amount

gc.collect()
historical_transactions.describe()
historical_installments = pd.DataFrame({'hist_total_installments': historical_transactions.groupby(['card_id'])['installments'].value_counts()}).reset_index()

historical_installments.head(5)
newM_installments = pd.DataFrame({'newM_total_installments': newMerchant_transactions.groupby(['card_id'])['installments'].value_counts()}).reset_index()

newM_installments.head(5)
train_df = pd.merge(train_df, historical_installments, on="card_id", how="left")

test_df = pd.merge(test_df, historical_installments, on="card_id", how="left")

train_df = pd.merge(train_df, newM_installments, on="card_id", how="left")

test_df = pd.merge(test_df, newM_installments, on="card_id", how="left")

train_df.head(5)
bins = pd.cut(train_df['installments_x'],[0,1,2,3,4,5,6,7,10,20,50,100,900,1000])

#as the highest number of transaction is 1137

hist_installments_data = train_df.groupby(bins)['installments_x'].agg(['count','sum','mean','std','min','max']).reset_index()

hist_installments_data
plt.figure(figsize=(18,5))

sns.boxplot(x=bins, y=train_df['target'], data=hist_installments_data, showfliers=False)

plt.xticks(rotation='90')

plt.ylabel('target/loyalt score')

plt.title('loyalt score based on number of installments in history transactions data')
bins = pd.cut(train_df['installments_y'],[0,1,2,3,4,5,6,7,10,20,50,100,900,1000])

#as the highest number of transaction is 1137

newM_installments_data = train_df.groupby(bins)['installments_y'].agg(['count','sum','mean','std','min','max']).reset_index()

newM_installments_data
plt.figure(figsize=(18,5))

sns.boxplot(x=bins, y=train_df['target'], data=newM_installments_data, showfliers=False)

plt.xticks(rotation='90')

plt.ylabel('target/loyalt score')

plt.title('loyalt score based on number of installments in new Merchant Data')
histPurchase_amount = pd.DataFrame({'hist_total_transactions' : historical_transactions.groupby(['card_id'])['purchase_amount'].size()}).reset_index()

histPurchase_amount.head(5)
newPurchase_amount = pd.DataFrame({'new_total_transactions' : newMerchant_transactions.groupby(['card_id'])['purchase_amount'].size()}).reset_index()

newPurchase_amount.head(5)
histPurchase_amount.columns = ['card_id','hist_total_transactions']

newPurchase_amount.columns = ['card_id','new_total_transactions']



train_df = pd.merge(train_df, histPurchase_amount, on = 'card_id', how = 'left')

test_df = pd.merge(test_df, histPurchase_amount, on = 'card_id', how = 'left')

train_df = pd.merge(train_df, newPurchase_amount, on = 'card_id', how = 'left')

test_df = pd.merge(test_df, newPurchase_amount, on = 'card_id', how = 'left')



train_df.head(5)
s1 = train_df.groupby(['hist_total_transactions'])['target'].mean()

s2 = train_df.groupby(['new_total_transactions'])['target'].mean()

plotly.offline.iplot({

    "data": [go.Scatter(x = s1.index, y = s1.values, name = 'hist_transactions'),

            go.Scatter(x = s2.index, y = s2.values, name = 'new_transactions')],

    "layout": go.Layout(title="Number of Transactions in History and New Data vs. Loyalt score based on Purchase amount")

})
bins = pd.cut(train_df['hist_total_transactions'],[0,10,20,30,40,50,60,70,80,100,120,150,300, 500,1000,1500,2000,3000])

#as the highest number of transaction is 1137

hist_data = train_df.groupby(bins)['hist_total_transactions'].agg(['count','sum','mean','std','min','max']).reset_index()

hist_data
hist_data.columns = ['card_id', 'count_hist_transaction', "sum_hist_transaction", "mean_hist_transaction", "std_hist_transaction",

                     "min_hist_transaction", "max_hist_transaction"]

train_df = pd.merge(train_df, hist_data, on="card_id", how="left")

test_df = pd.merge(test_df, hist_data, on="card_id", how="left")
plt.figure(figsize=(18,5))

sns.boxplot(x=bins, y=train_df['target'], data=train_df)

plt.xticks(rotation='90')

plt.ylabel('target/loyalt score')

plt.title('loyalt score based on number of purchased amount in new history Data')
(train_df.loc[train_df['hist_total_transactions'] > 1500]).head()
bins = pd.cut(train_df['hist_total_transactions'],[1000,1500,2000,3000])

#as the highest number of transaction is 1137

hist_data = train_df.groupby(bins)['hist_total_transactions'].agg(['mean']).reset_index()

plt.figure(figsize=(15,5))

sns.boxplot(x=bins, y=train_df['target'], data=train_df, showfliers=False)

plt.xticks(rotation='90')

plt.ylabel('target/loyalt score')

plt.title('loyalt score based on purchase amount')
(train_df.loc[(train_df['first_active_month'] == '2017-01')]).head()
historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'], infer_datetime_format=False)
historical_transactions['month_yr'] = historical_transactions['purchase_date'].apply(lambda x: x.strftime('%B-%Y'))

historical_transactions.head(5)
history_repeat_business = pd.DataFrame({'hist_transactions' : historical_transactions.groupby('card_id')['month_yr'].value_counts()}).reset_index()

history_repeat_business.head(5)
bins1 = pd.cut(history_repeat_business['hist_transactions'],[0,10,20,30,40,50,60,70,80,100,120,150,1000,1500])

#as the highest number of transaction is 1137

hist_data = history_repeat_business.groupby(bins1)['hist_transactions'].agg(['count','sum'])

hist_data
plt.figure(figsize=(15,5))

sns.countplot(x = bins1, data = hist_data)
cnt_srs = history_repeat_business['month_yr'].value_counts()

cnt_srs
#month_year data sorting.

month_yr = ["February-2018","January-2018","January-2017", "February-2017", "March-2017", "April-2017", "May-2017", "June-2017", 

          "July-2017", "August-2017", "September-2017", "October-2017", "November-2017", "December-2017"]

history_repeat_business['month_yr'] = pd.Categorical(history_repeat_business['month_yr'], categories=month_yr, ordered=True)

history_repeat_business.sort_values(by='month_yr')
plotly.offline.iplot({

    "data": [go.Scatter(x = cnt_srs.index[::-1], y = cnt_srs.values[::-1], name = 'transactions')],

    "layout": go.Layout(title="Number of transactions in year 2017")

})
#history_repeat_business = pd.DataFrame({'hist_transactions' : historical_transactions.groupby('card_id')['month_yr'].value_counts()}).reset_index()

history_repeat_business.loc[history_repeat_business['hist_transactions'] >= 500]
'''L = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter']

hist_date_data = historical_transactions.join(pd.concat((getattr(historical_transactions['purchase_date'].dt, i).rename(i) for i in L), axis=1))

#new_date_data = new_date_data.join(pd.concat((getattr(new_date_data['purchase_date'].dt, i).rename(i) for i in L), axis=1))

hist_date_data.head(5)'''
'''months_map = {1: 'Jan', 2: 'Feb', 3:'March', 4:'Apr', 5:'May', 6:'June', 7:'July', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4:'Friday', 5:'Saturday', 6: 'Sunday'}

hist_date_data['month'] = hist_date_data['month'].apply(lambda x: months_map[x])

hist_date_data['dayofweek'] = hist_date_data['dayofweek'].apply(lambda x: day_map[x])

hist_day_data = pd.DataFrame({'hist_transactions' : hist_date_data.groupby('card_id')['dayofweek'].value_counts()}).reset_index()



#day_srs = hist_date_data.groupby('card_id')['dayofweek'].value_counts()

hist_day_data.head(5)'''
'''new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")

new_trans_df.head()'''
'''newMerchant_transactions = reduce_mem_usage(new_trans_df)

gc.collect()'''
newMerchant_transactions['purchase_date'] = pd.to_datetime(newMerchant_transactions['purchase_date'], infer_datetime_format=False)
newMerchant_transactions['month_yr'] = newMerchant_transactions['purchase_date'].apply(lambda x: x.strftime('%B-%Y'))

newMerchant_transactions.head(5)
newMerchant_transactions = pd.DataFrame({'new_merchant_transactions' : newMerchant_transactions.groupby('card_id')['month_yr'].size()}).reset_index()

newMerchant_transactions.head(5)
train_df = pd.merge(train_df, newMerchant_transactions, on = 'card_id', how = 'left')

test_df = pd.merge(test_df, newMerchant_transactions, on = 'card_id', how = 'left')
train_df.head(5)
bins = pd.cut(newMerchant_transactions['new_merchant_transactions'],[0,10,20,30,40,50,60,70,80])

#as the highest number of transaction is 1137

newM_data = newMerchant_transactions.groupby(bins)['new_merchant_transactions'].agg(['count','sum','mean','std','min','max']).reset_index()

newM_data
bins = pd.cut(newMerchant_transactions['new_merchant_transactions'],[0,10,20,30,40,50,60,70,80])

#as the highest number of transaction is 1137

newM_data = newMerchant_transactions.groupby(bins)['new_merchant_transactions'].agg(['count','sum','mean','std','min','max']).reset_index()

newM_data
newM_data.columns = ["card_id", "count_newM_trans", "sum_newM_trans", "mean_newM_trans", "std_newM_trans", "min_newM_trans", "max_newM_trans"]

train_df = pd.merge(train_df, newM_data, on="card_id", how="left")

test_df = pd.merge(test_df, newM_data, on="card_id", how="left")
train_df.head(5)
plt.figure(figsize=(15,5))

sns.countplot(x = bins, data = train_df)

plt.ylabel('target/loyalt score')
bins = pd.cut(train_df['new_merchant_transactions'],[0,10,20,30,40,50,60,70,80,100,120])

newM_srs = train_df.groupby(bins)['target'].value_counts()

plt.figure(figsize=(15,5))

sns.boxplot(x=bins, y='target', data=train_df, showfliers=False)

plt.ylabel('target/loyalt score')

plt.xticks(rotation = '90')

plt.title('new Merchant transaction distribution by target/loyalty score values')

plt.show()
train_df['first_active_month'] = pd.to_datetime(train_df['first_active_month'], infer_datetime_format=False)
test_df['first_active_month'] = pd.to_datetime(test_df['first_active_month'], infer_datetime_format=False)
train_df['year'] = train_df['first_active_month'].dt.year

test_df['year'] = test_df['first_active_month'].dt.year

train_df['month'] = train_df['first_active_month'].dt.year

test_df['month'] = test_df['first_active_month'].dt.month
cols = ['feature_1', 'feature_2', 'feature_3','installments_x', 'hist_total_installments', 'installments_y',

       'newM_total_installments', 'hist_total_transactions',

       'new_total_transactions', 'count_hist_transaction',

       'sum_hist_transaction', 'mean_hist_transaction', 'std_hist_transaction',

       'min_hist_transaction', 'max_hist_transaction',

       'new_merchant_transactions', 'count_newM_trans', 'sum_newM_trans',

       'mean_newM_trans', 'std_newM_trans', 'min_newM_trans', 'max_newM_trans',

       'year', 'month']
from sklearn import model_selection, preprocessing, metrics

import lightgbm as lgb



x_train = train_df[cols]

x_test = test_df[cols]

y_train = train_df[cols].values

y_test = test_df[cols].values



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



#---Build LGB Model-----

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)







def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 30,

        "min_child_weight" : 50,

        "learning_rate" : 0.05,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.7,

        "bagging_frequency" : 5,

        "bagging_seed" : 2018,

        "verbosity" : -1,

        'max_depth' : 10,

        'n_estimators' : 200,

        'min_child_samples': 399, 

        'min_child_weight': 0.1

    }

    

    train_data=lgb.Dataset(train_X,label=train_y)

    valid_data=lgb.Dataset(val_X,label=val_y)

 

    evals_result = {}

    model = lgb.train(params, train_data, 1000, valid_sets=[valid_data], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)

    

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result



train_X = train_df[cols]

test_X = test_df[cols]

train_y = train_df['target'].values



pred_test = 0

kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)

for dev_index, val_index in kf.split(train_df):

    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

    pred_test += pred_test_tmp

pred_test /= 5.
fig, ax = plt.subplots(figsize=(12,10))

lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()
sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})

sub_df["target"] = pred_test

sub_df.to_csv("lgb_baseline.csv", index=False)