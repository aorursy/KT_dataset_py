# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#df_merchants=pd.read_csv("../input/merchants.csv")

df_historical_transactions=pd.read_csv("../input/historical_transactions.csv")

df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")
df_train.groupby('first_active_month').count()['card_id'].plot(kind='bar',figsize=(40,15))
df_test.groupby('first_active_month').count()['card_id'].plot(kind='bar',figsize=(40,15))
df_train['target'].describe()
df_train.boxplot(column='target', figsize=(20,20))
df_train['target'].describe()['75%'] - df_train['target'].describe()['25%']
df_train.head()
df_train.sort_values('first_active_month').groupby(['first_active_month']).mean()['target'].plot(kind='bar',figsize=(20,10))

fig, ax = plt.pyplot.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

for feature in df_train['feature_1'].unique():

        sns.distplot(df_train[df_train['feature_1']==feature]['target']);
fig, ax = plt.pyplot.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns.violinplot(x="feature_1", y="target",  data=df_train, palette="muted",inner="points")
fig, ax = plt.pyplot.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns.violinplot(x="feature_2", y="target",  data=df_train, palette="muted",inner="points")
fig, ax = plt.pyplot.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns.violinplot(x="feature_3", y="target",  data=df_train, palette="muted",inner="points")
df_historical_transactions.head()
print("min purchase date", df_historical_transactions['purchase_date'].min())

print("max purchase date", df_historical_transactions['purchase_date'].max())
df_historical_transactions['purchase_date'] = pd.to_datetime(df_historical_transactions['purchase_date'])

max_purchase_date=df_historical_transactions['purchase_date'].max()
g=df_historical_transactions[['card_id']].groupby('card_id')
df_transaction_counts=g.size().reset_index(name='num_transactions')
#df_transaction_counts.head()

df_train=pd.merge(df_train,df_transaction_counts, on="card_id",how='left')

df_test=pd.merge(df_test,df_transaction_counts, on="card_id",how='left')
df_train['num_transactions'].describe()
bins=[0,500,1000,1500,2000,2500]
df_train['binned_numtransactions']=pd.cut(df_train['num_transactions'],bins)
df_train.head()
fig, ax = plt.pyplot.subplots()

# the size of A4 paper

fig.set_size_inches(11.7, 8.27)

sns.boxplot(x='binned_numtransactions',y='target',data=df_train)
#this takes more memory (not needed now)

#df_historical_transactions['YearMonth_Purchase'] = df_historical_transactions['purchase_date'].map(lambda x: 100*x.year + x.month)



#You can plot and check this , there is peak in one particular month, skipping this as it takes memory

#g=df_historical_transactions[['YearMonth_Purchase','purchase_amount']].groupby('YearMonth_Purchase').mean()

#g.plot(kind='bar',figsize=(20,10))
top_merchants_by_purchaseamount=df_historical_transactions[['merchant_id','purchase_amount']].groupby(by='merchant_id').mean().sort_values(by='purchase_amount',ascending=False).head(20)

top_merchants_by_purchaseamount.head()
 

#g=df_historical_transactions[df_historical_transactions['merchant_id'].isin(list(top_merchants_by_purchaseamount.index))][['merchant_id','YearMonth_Purchase','purchase_amount']].groupby(['YearMonth_Purchase','merchant_id']).mean()

#g.unstack()
g=df_historical_transactions[['card_id','merchant_id']].groupby(['card_id','merchant_id'])

merchantid_counts_percard=g.size()

merchantid_counts_percard=pd.DataFrame(merchantid_counts_percard)
merchantid_counts_percard.head()
merchantid_counts_percard.columns=['num_favourite_merchant']
merchantid_counts_percard.head()
merchantid_counts_percard=merchantid_counts_percard.sort_values(by='num_favourite_merchant',ascending=False)
merchantid_counts_percard=merchantid_counts_percard.groupby(level=0).head(1).reset_index()

merchantid_counts_percard.columns=['card_id','favourite_merchant','num_transaction_favourite_merchant']


df_train=pd.merge( df_train,merchantid_counts_percard,on="card_id",how='left')
df_test=pd.merge( df_test,merchantid_counts_percard,on="card_id",how='left')
df_historical_transactions['purchase_amount'].describe()
g=df_historical_transactions[['card_id','purchase_amount']].groupby('card_id')
purchaseamount_agg=g.agg(['sum', 'min','max','std','median','mean'])
purchaseamount_agg=purchaseamount_agg.reset_index()
purchaseamount_agg.head()
df_train=pd.merge( df_train,purchaseamount_agg,on="card_id",how='left')
df_train.head()
df_test=pd.merge( df_test,purchaseamount_agg,on="card_id",how='left')
df_train["first_active_month"]=pd.to_datetime(df_train["first_active_month"])
df_test["first_active_month"]=pd.to_datetime(df_test["first_active_month"])
df_train["first_active_yr"]=df_train["first_active_month"].dt.year
df_test["first_active_yr"]=df_test["first_active_month"].dt.year
df_train["first_active_mon"]=df_train["first_active_month"].dt.month
df_test["first_active_mon"]=df_test["first_active_month"].dt.month
len(df_train['favourite_merchant'].unique())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['favourite_merchant'] = le.fit_transform(df_train['favourite_merchant'] )
le = preprocessing.LabelEncoder()

df_test['favourite_merchant'] = le.fit_transform(df_test['favourite_merchant'] )
last_active_month=df_historical_transactions.loc[df_historical_transactions.groupby('card_id').purchase_date.idxmax(),:][['card_id','purchase_date','purchase_amount']]

last_active_month.columns=['card_id','last_active_purchase_date','last_active_purchase_amount']
last_active_month.head()
df_train=pd.merge(df_train,last_active_month, on="card_id",how='left')

df_test=pd.merge(df_test,last_active_month, on="card_id",how='left')

df_train['last_active_purchase_year']=df_train['last_active_purchase_date'].dt.year
df_train['last_active_purchase_month']=df_train['last_active_purchase_date'].dt.month
df_train['last_active_purchase_day']=df_train['last_active_purchase_date'].dt.day
df_test['last_active_purchase_year']=df_test['last_active_purchase_date'].dt.year
df_test['last_active_purchase_month']=df_test['last_active_purchase_date'].dt.month
df_test['last_active_purchase_day']=df_test['last_active_purchase_date'].dt.day
max_purchase_date


df_train['dormancy']=[(max_purchase_date-x).days for x in df_train['last_active_purchase_date']]
df_test['dormancy']=[(max_purchase_date-x).days for x in df_test['last_active_purchase_date']]
df_train.head()
df_test.columns




from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import lightgbm as lgb
df_train.columns=['first_active_month','card_id','feature_1','feature_2','feature_3','target','num_transactions',

                  'binned_numtransactions','favourite_merchant','num_transaction_favourite_merchant',

                  'sum_purchase_amount','min_purchase_amount','max_purchase_amount',

                  'std_purchase_amount','median_purchase_amount','mean_purchase_amount',

                  'first_active_yr','first_active_mon','last_active_purchase_date',

       'last_active_purchase_amount', 'last_active_purchase_year','last_active_purchase_month', 

                  'last_active_purchase_day', 'dormancy']

df_test.columns=['first_active_month','card_id','feature_1','feature_2','feature_3',

                 'num_transactions','favourite_merchant','num_transaction_favourite_merchant',

                 'sum_purchase_amount','min_purchase_amount','max_purchase_amount',

                 'std_purchase_amount','median_purchase_amount','mean_purchase_amount','first_active_yr','first_active_mon','last_active_purchase_date',

            'last_active_purchase_amount', 'last_active_purchase_year',

           'last_active_purchase_month', 'last_active_purchase_day', 'dormancy']

df_test.head()
final_cols=['feature_1','feature_2','feature_3','num_transactions','favourite_merchant','num_transaction_favourite_merchant','sum_purchase_amount',

            'min_purchase_amount','max_purchase_amount','std_purchase_amount','median_purchase_amount','mean_purchase_amount',

            'first_active_yr','first_active_mon','last_active_purchase_amount', 'last_active_purchase_year',

           'last_active_purchase_month', 'last_active_purchase_day', 'dormancy']

target_col=['target']


lgb_params = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 30,

        "min_child_weight" : 50,

        "learning_rate" : 0.05,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.7,

        "bagging_frequency" : 5,

        "bagging_seed" : 2018,

        "verbosity" : -1

    }
Folds = KFold(n_splits=7, shuffle=True, random_state=1989)


pred_train = np.zeros(len(df_train))

pred_test = np.zeros(len(df_test))



features_lgb = list(df_train.columns)

feature_importance = pd.DataFrame()
train_X=df_train[final_cols]
train_y=df_train[target_col]
test_X=df_test[final_cols]
for fold_, (train_idx, val_idx) in enumerate(Folds.split(df_train)):

    train_data = lgb.Dataset(train_X.iloc[train_idx], label=train_y.iloc[train_idx])

    val_data = lgb.Dataset(train_X.iloc[val_idx], label=train_y.iloc[val_idx])



    num_round = 10000

    model = lgb.train(lgb_params, train_data, num_round, valid_sets = [train_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)

    pred_train[val_idx] = model.predict(train_X.iloc[val_idx], num_iteration=model.best_iteration)



    pred_test += model.predict(test_X, num_iteration=model.best_iteration) / Folds.n_splits





print(np.sqrt(mean_squared_error(pred_train, df_train[target_col])))
submit_df = pd.read_csv('../input/sample_submission.csv')

submit_df["target"] = pred_test

submit_df.to_csv("submission_baseline_lgb.csv", index=False)
submit_df
fig, ax = plt.pyplot.subplots(figsize=(12,10))

lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.pyplot.title("LGB- Feature Importance", fontsize=15)
