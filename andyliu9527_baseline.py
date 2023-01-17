# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import gc

import os
tmp_dir='../input'
print(os.listdir(tmp_dir))


# Any results you write to the current directory are saved as output.
train=pd.read_csv(tmp_dir+'/train.csv')
test=pd.read_csv(tmp_dir+'/test.csv')
col_describe=pd.read_excel(tmp_dir+'/Data_Dictionary.xlsx')
print(train.shape)
print(test.shape)
train.head()
train.isnull().sum()
test.isnull().sum()
for i in [ 'feature_1', 'feature_2', 'feature_3']:
    train[i]=train[i].astype('category')
    test[i]=test[i].astype('category')
train.describe(include='all')
test.describe()
sns.distplot(train.target)
train.target.value_counts().sort_index()
train['target2']=train.target.map(lambda x:round(x,3))
train=train[train.target2!=-33.219]
del train['target2']
### fill null 
a=test.first_active_month.value_counts()
test.first_active_month.fillna(a[a.values==a.max()].index[0],inplace=True)
print(train.first_active_month.min())
print(train.first_active_month.max())
print(test.first_active_month.min())
print(test.first_active_month.max())

train.first_active_month.describe()

## count by first_active_month
test_month_count=test.first_active_month.value_counts().sort_index()
train_month_count=train.first_active_month.value_counts().sort_index()

plt.plot(test_month_count.index,test_month_count.values)
plt.plot(train_month_count.index,train_month_count.values)
## mean of target by first_active_month

mean_target=train.groupby('first_active_month')['target'].mean().sort_index()
plt.plot(mean_target.index,mean_target.values)
y_true=train.target
del train['target']
train_count=train.shape[0]   
df=pd.concat([train,test],axis=0) ##train=df.iloc[:train_count]
del train
del test
df.first_active_month=df.first_active_month.astype('category')
df_dummy=pd.get_dummies(df[['feature_1','feature_2','feature_3','first_active_month']])
df=pd.concat([df,df_dummy],axis=1)
del df_dummy

df['first_active_month']=pd.to_datetime(df['first_active_month'])
df['first_month']=df.first_active_month.dt.month
max_date=df.first_active_month.dt.date.max()
df['lantency']=max_date - df.first_active_month.dt.date
df['lantency']=df.lantency.dt.days

df=df.drop(['feature_1', 'feature_2','feature_3','first_active_month'], axis=1)
df.first_month=df.first_month.astype('int8')
df.lantency=df.lantency.astype('int8')
gc.collect()
df.head()

### merchants.csv - additional information about all merchants / merchant_ids in the dataset.
# merchants=pd.read_csv(tmp_dir+'/merchants.csv')
# print(merchants.shape)
# merchants.head()
# a=merchants.merchant_id.value_counts()
# merchants2=merchants.groupby('merchant_id').head(1)
# merchants2.head()
##new_merchant_transactions.csv - two months' worth of data for each card_id 
###containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.

new_merchant_transactions=pd.read_csv(tmp_dir+'/new_merchant_transactions.csv')
print(new_merchant_transactions.shape)
for i in [ 'city_id','category_1','merchant_category_id','category_2','state_id',
          'subsector_id','authorized_flag','category_3','merchant_id']:
    new_merchant_transactions[i]=new_merchant_transactions[i].astype('category')
new_merchant_transactions.month_lag=new_merchant_transactions.month_lag.astype('int8')
new_merchant_transactions.installments=new_merchant_transactions.installments.astype('int8')
new_merchant_transactions.purchase_amount=new_merchant_transactions.purchase_amount.astype('float32')

print(new_merchant_transactions.dtypes)
new_merchant_transactions.head()
# new_merchant_transactions.head()
# new_merchant_transactions.describe(include='category')
# new_merchant_transactions.describe()
# new_merchant_transactions.isnull().sum()
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(new_merchant_transactions['category_1'])
new_merchant_transactions['category_1']=le.transform(new_merchant_transactions['category_1'])
new_merchant_transactions['category_1']=new_merchant_transactions['category_1'].astype('int8')
new_merchant_transactions['purchase_amount']=new_merchant_transactions['purchase_amount'].astype('float32')

 
tmp=pd.get_dummies(new_merchant_transactions[['category_2','category_3']])
new_merchant_transactions=pd.concat([new_merchant_transactions,tmp],axis=1)

new_merchant_transactions.purchase_date=pd.to_datetime(new_merchant_transactions.purchase_date)
for i in ['hour','day','dayofweek','month']:
    new_merchant_transactions['purchase_'+i]=getattr(new_merchant_transactions.purchase_date.dt,i)
    new_merchant_transactions['purchase_'+i]=new_merchant_transactions['purchase_'+i].astype('int8')
new_merchant_transactions.head()
%%time
    
agg_fun={'category_1': ['sum', 'mean'],
'category_2_1.0': ['sum', 'mean'],
'category_2_2.0': ['sum', 'mean'],
'category_2_3.0': ['sum', 'mean'],
'category_2_4.0': ['sum', 'mean'],
'category_2_5.0': ['sum', 'mean'],
'category_3_A': ['sum', 'mean'],
'category_3_B': ['sum', 'mean'],
'category_3_C': ['sum', 'mean'],
'month_lag': ['sum', 'mean'],
'installments': ['sum', 'mean','min','max','std'],
'purchase_amount': ['sum', 'mean','min','max','std'],
        
'state_id':['nunique'], 
'city_id': ['nunique'],
'merchant_category_id': ['nunique'],
'subsector_id': ['nunique'],
'merchant_id': ['nunique'],
         
'purchase_hour': ['min', 'max'],
'purchase_day': ['min', 'max'],
'purchase_dayofweek': ['min', 'max'],
'purchase_month': ['min', 'max'],

'card_id':'count',
}

new_agg=new_merchant_transactions.groupby('card_id').agg(agg_fun)

new_agg.columns=['new_' + '_'.join(col).strip() for col in new_agg.columns.values]
del new_merchant_transactions
gc.collect()
# new_agg.to_csv('new_agg.csv')
%%time
#### historical_transactions.csv - up to 3 months' worth of historical transactions for each card_id

historical_transactions=pd.read_csv(tmp_dir+'/historical_transactions.csv')
print(historical_transactions.shape)
for i in [ 'city_id','category_1','merchant_category_id','category_2','state_id',
          'subsector_id','authorized_flag','category_3','merchant_id']:
    historical_transactions[i]=historical_transactions[i].astype('category')
historical_transactions['month_lag']=historical_transactions['month_lag'].astype('int8')
historical_transactions.installments=historical_transactions.installments.astype('int8')
historical_transactions.purchase_amount=historical_transactions.purchase_amount.astype('float32')
print(historical_transactions.dtypes)
historical_transactions.head()
le=preprocessing.LabelEncoder()
le.fit(historical_transactions['category_1'])
historical_transactions['category_1']=le.transform(historical_transactions['category_1'])

tmp=pd.get_dummies(historical_transactions[['category_2','category_3']])
historical_transactions=pd.concat([historical_transactions,tmp],axis=1)

historical_transactions.purchase_date=pd.to_datetime(historical_transactions.purchase_date)
for i in ['hour','day','dayofweek','month']:
    historical_transactions['purchase_'+i]=getattr(historical_transactions.purchase_date.dt,i)
    historical_transactions['purchase_'+i]=historical_transactions['purchase_'+i].astype('int8')
historical_transactions.head()
%%time
    
agg_fun={'category_1': ['sum', 'mean'],
'category_2_1.0': ['sum', 'mean'],
'category_2_2.0': ['sum', 'mean'],
'category_2_3.0': ['sum', 'mean'],
'category_2_4.0': ['sum', 'mean'],
'category_2_5.0': ['sum', 'mean'],
'category_3_A': ['sum', 'mean'],
'category_3_B': ['sum', 'mean'],
'category_3_C': ['sum', 'mean'],
'month_lag': ['sum', 'mean'],
'installments': ['sum', 'mean','min','max','std'],
'purchase_amount': ['sum', 'mean','min','max','std'],
        
'state_id':['nunique'], 
'city_id': ['nunique'],
'merchant_category_id': ['nunique'],
'subsector_id': ['nunique'],
'merchant_id': ['nunique'],
         
'purchase_hour': ['min', 'max'],
'purchase_day': ['min', 'max'],
'purchase_dayofweek': ['min', 'max'],
'purchase_month': ['min', 'max'],

'card_id':'count',
}

his_agg=historical_transactions.groupby('card_id').agg(agg_fun)
his_agg.columns=['his_' + '_'.join(col).strip() for col in his_agg.columns.values]
del historical_transactions
gc.collect()
# his_agg.to_csv('new_agg.csv')
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
df=pd.merge(df,new_agg,left_on='card_id',right_index=True,how='left')
print(len(df.columns))
df=pd.merge(df,his_agg,left_on='card_id',right_index=True,how='left')
print(len(df.columns))

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


df=reduce_mem_usage(df)
df=df.set_index('card_id')

del new_agg
del his_agg
df_train=df.iloc[:train_count]
df_test=df.iloc[train_count:]
print('df_train.shape',df_train.shape,'df_test.shape',df_test.shape)
del df
gc.collect()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(df_train, y_true, test_size=0.2, random_state=8)
# del df_train
%%time
## trian by raw lightgbm
from sklearn.metrics import mean_squared_error
train_columns=X_train.columns.tolist()
train_data = lgb.Dataset(X_train, label=y_train,feature_name=train_columns
                        ,free_raw_data=False)

param = {'boosting_type':['gbdt'],'max_leaves':[33], 'min_data_in_leaf':[30],'max_depth':[5],
         'objective':['regression'],'random_state':[8],'metric':['l2']}
num_round=100
bst=lgb.train(param,train_data,num_round)
mean_squared_error(y_test,bst.predict(X_test))

## train by sklearn GridSearchCV

# from sklearn.metrics import mean_squared_error


# param = {'boosting_type':['gbdt'],'max_leaves':[33], 'min_data_in_leaf':[30,40,50],'max_depth':[4,5,6],
#          'objective':['regression'],'random_state':[8],'metric':['l2']}

# clf=GridSearchCV(lgb.LGBMRegressor(),param_grid=param,cv=5,n_jobs=4,verbose=2)
# clf.fit(X_train, y_train,feature_name=['first_active_month', 'feature_1', 'feature_2', 'feature_3'],
#         categorical_feature=['first_active_month', 'feature_1', 'feature_2', 'feature_3'])

# mean_squared_error(y_test,clf.best_estimator_.predict(X_test))
df_test.head()
## predict by raw lightgbm
predict_target=bst.predict(df_test)

## predict  sklearn GridSearchCV
# predict_target=clf.best_estimator_.predict(X_predict)

df_test['target']=predict_target
df_test=df_test.reset_index()
df_test[['card_id','target']].to_csv('submission.csv',index=False)
print('success asv to csv submission.csv')
