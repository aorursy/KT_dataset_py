import numpy as np 

import pandas as pd

import os

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import pickle

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

import time
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
sales_train['date']=pd.to_datetime(sales_train['date'],format='%d.%m.%Y')



sales_train=sales_train.assign(day=sales_train['date'].dt.day,

                  month=sales_train['date'].dt.month,

                  year=sales_train['date'].dt.year)
def data_cleaner(df):

    ##selecting non-string cols

    X=df[['date_block_num','shop_id','item_id','item_price']]

    y=df['item_cnt_day']



    ##grouping by per_month, per_shop, per_item

    all_data=X.join(y)

    a=all_data.groupby(['date_block_num','shop_id','item_id']).sum()

    a['item_price']=all_data.groupby(['date_block_num','shop_id','item_id'])['item_price'].mean()



    ##breaking GroupBy multiIndex and renaming the groupby-summed column

    a.reset_index(inplace=True)

    a.rename({'item_cnt_day':'item_cnt_month'},inplace=True,axis=1)

    all_data=a



    ##redefining X and y

    X=all_data.drop('item_cnt_month',axis=1)

    y=all_data['item_cnt_month']



    ##Mapping item_categories_id to all_data

    all_data['item_category_id']=all_data['item_id'].map(items['item_category_id'])



    ##rearranging all_data column sequence

    all_data=all_data[['date_block_num','shop_id','item_id','item_category_id','item_price','item_cnt_month']]

    

    return all_data
def model_evaluator(ad):

    model=LinearRegression()

    curr_block_num=33

    X_train = ad.loc[ad['date_block_num']<33].drop(['item_cnt_month'],axis=1)

    y_train = ad.loc[ad['date_block_num']<33]['item_cnt_month']

    X_valid = ad.loc[ad['date_block_num']==33].drop(['item_cnt_month'],axis=1)

    y_valid = ad.loc[ad['date_block_num']==33]['item_cnt_month']

    model.fit(X_train,y_train)

    preds_valid=model.predict(X_valid)

    

    print(np.sqrt(mean_squared_error(y_valid,preds_valid)))

    print('r2 error',r2_score(y_valid,preds_valid))

    return model
def model_evaluator2(ad):

    model=RandomForestRegressor(max_depth=7,n_estimators=200,n_jobs=-1,max_samples=0.8, max_features=0.8)

    curr_block_num=33

    X_train = ad.loc[ad['date_block_num']<curr_block_num].drop(['item_cnt_month'],axis=1)

    y_train = ad.loc[ad['date_block_num']<curr_block_num]['item_cnt_month']

    X_valid = ad.loc[ad['date_block_num']==curr_block_num].drop(['item_cnt_month'],axis=1)

    y_valid = ad.loc[ad['date_block_num']==curr_block_num]['item_cnt_month']

    model.fit(X_train,y_train)

    preds_valid=model.predict(X_valid)

    

    print('rmse for validation data',np.sqrt(mean_squared_error(y_valid,preds_valid)))

    print('r2 error',r2_score(y_valid,preds_valid))

    return model
def rf_evaluator(ad,depth):

    model=RandomForestRegressor(max_depth=depth,n_estimators=50,n_jobs=-1,max_samples=0.8, max_features=0.8)

    curr_block_num=33

    X_train = ad.loc[ad['date_block_num']<curr_block_num].drop(['item_cnt_month'],axis=1)

    y_train = ad.loc[ad['date_block_num']<curr_block_num]['item_cnt_month']

    X_valid = ad.loc[ad['date_block_num']==curr_block_num].drop(['item_cnt_month'],axis=1)

    y_valid = ad.loc[ad['date_block_num']==curr_block_num]['item_cnt_month']

    model.fit(X_train,y_train)

    preds_valid=model.predict(X_valid)

    

    print('rmse for validation data',np.sqrt(mean_squared_error(y_valid,preds_valid)))

    print('r2 error',r2_score(y_valid,preds_valid))

    return model
all_data=data_cleaner(sales_train)

model_evaluator(all_data)

all_data.head()
all_data.isnull().any()
fig=plt.figure(figsize=(10,5))



a=all_data.groupby(['date_block_num'])['item_cnt_month'].sum()

plt.plot(a.index,a.values)

plt.xlabel('date_block_num')

plt.ylabel('item_cnt_month')
g=sns.FacetGrid(sales_train)

g.map(sns.boxplot,x=sales_train['item_price'])
print('max item price',sales_train['item_price'].max())

print('min item price',sales_train['item_price'].min())
max_id=sales_train['item_price'].idxmax()

min_id=sales_train['item_price'].idxmin()



##dropping the outliers

a=sales_train.drop([max_id,min_id])

all_data=data_cleaner(a)



#rechecking distribuion

print('min item price',a.item_price.min())

print('max item price',a.item_price.max())

sns.boxplot(a.item_price)
a.loc[a.item_price>40000].shape
a1=a.drop(a.loc[a.item_price>40000].index,axis=0,inplace=False)

all_data=data_cleaner(a1)
plt.figure(figsize=(10,4))

sns.scatterplot(all_data.index,all_data['item_cnt_month'].values)

plt.xlabel('index')

plt.ylabel('item_cnt_month')
plt.figure(figsize=(10,4))

plt.plot(all_data.index,all_data['item_cnt_month'].values)

plt.ylabel('item_cnt_month')

plt.xlabel('index')
g=sns.FacetGrid(all_data,height=4, aspect=3)

g.map(sns.boxplot,x=all_data['item_cnt_month'])

## it seems there are still some outliers greater than 1000
sns.set_style(style='whitegrid')

g=sns.FacetGrid(all_data,height=4, aspect=3)

g.map(sns.boxplot,x=all_data['shop_id'],y=all_data['item_cnt_month'])

## hence we can exclude data with monthly sales count more than 1000
all_data=all_data.loc[all_data['item_cnt_month']<=1000]



##rechecking for outliers

g=sns.FacetGrid(all_data,height=4, aspect=3)

g.map(sns.boxplot,x=all_data['item_cnt_month'])
sns.distplot(all_data['item_category_id'])

## found no outliers
## outlier exist when grouped by and summed

a=all_data.groupby('item_id').sum()

sns.boxplot(a.item_cnt_month)
sns.boxplot(all_data.loc[all_data['item_id']==20949]['item_cnt_month'])
all_data.to_pickle('cleaned_numeric1.pkl')

all_data=pd.read_pickle('cleaned_numeric1.pkl')
a=all_data.groupby('item_category_id').sum()['item_cnt_month']

sns.pointplot(a.index,a.values)
a=all_data.groupby('shop_id').sum()['item_cnt_month']

sns.pointplot(a.index,a.values)
index_cols=['shop_id','item_id','date_block_num']
all_data=pd.read_pickle('cleaned_numeric1.pkl')

def train_val_splitter(df):

    val_data=df.loc[df['date_block_num']==33]

    train_data=df.loc[df['date_block_num']<33]

    return train_data,val_data
## encoding done on train set and 

train_data,val_data=train_val_splitter(all_data[['shop_id','item_id','date_block_num','item_cnt_month']])

a1=train_data.groupby('item_id').mean()['item_cnt_month']

all_data['item_id_mean']=all_data['item_id'].map(a1)

fillnan=all_data.item_id_mean.mean()

all_data.fillna(value=fillnan,inplace=True)

np.corrcoef(all_data['item_cnt_month'],all_data['item_id_mean'])[0][1]
## RMSE for unregularized mean encoding

model_evaluator(all_data)
def CV_encoder(df):

    for train_index,val_index in kf.split(df):

        train=df.iloc[train_index]

        val=df.iloc[val_index]

        mean=train.groupby(df.columns[0]).mean()['item_cnt_month']

        train_new.iloc[val_index]=val[df.columns[0]].map(mean)

    train_new.fillna(value=df['item_cnt_month'].mean(),inplace=True)

    return train_new
all_data=pd.read_pickle('cleaned_numeric1.pkl')

from sklearn.model_selection import KFold

kf=KFold(n_splits=5)

global_mean=all_data['item_cnt_month'].mean()



columns_to_encode=['item_id','shop_id','item_category_id']

train_new=pd.Series(data=None,index=all_data.index)

all_data['item_cnt_month']=all_data['item_cnt_month'].clip(0,20)



#iterating through each column, encoding them and adding them to the dataframe

for i in np.arange(0,len(columns_to_encode)):

    a=all_data[[columns_to_encode[i],'item_cnt_month']]

    train_new=CV_encoder(a)

    all_data[a.columns[0]+'_cv_enc']=train_new

    print(i)
model_evaluator(all_data.drop(columns_to_encode,axis=1))
all_data.to_pickle('cleaned_numeric2.pkl')

all_data=pd.read_pickle('cleaned_numeric2.pkl')

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv",index_col='ID')
## clipping all_data targets from 0,20

all_data['item_cnt_month']=all_data['item_cnt_month'].clip(0,20)
## mapping item category id to test

test['item_category_id']=test['item_id'].map(items.set_index('item_id')['item_category_id'])



##adding the mean item_price to all shop_id/item_id combinations

a=sales_train.groupby(['shop_id','item_id']).mean().reset_index().drop(['date_block_num','item_cnt_day'],axis=1)

test=pd.merge(test,a,on=['shop_id','item_id'],how='left')
##mapping item_id mean encoding for test data

a=all_data.groupby('item_id')['item_cnt_month'].mean()

test['item_id_cv_enc']=test['item_id'].map(a)



##mapping shop_id mean encoding for test data

a=all_data.groupby('shop_id')['item_cnt_month'].mean()

test['shop_id_cv_enc']=test['shop_id'].map(a)



##mapping item_category_id mean encoding for test data

a=all_data.groupby('item_category_id')['item_cnt_month'].mean()

test['item_category_id_cv_enc']=test['item_category_id'].map(a)



##rearranging columns

test['date_block_num']=34

test=test[all_data.columns.drop('item_cnt_month')]

test.fillna(0,inplace=True)
lr=model_evaluator(all_data)

test_preds=lr.predict(test)

test_preds=np.round(test_preds,decimals=0).astype(int)



sample_submission_copy=sample_submission

sample_submission_copy['item_cnt_month']=test_preds

sample_submission_copy.to_csv('lr_no_text.csv',index=False)

pd.read_csv('lr_no_text.csv')
test.to_pickle('test1.pkl')
all_data.head()
shops['city']=shops['shop_name'].map(lambda x: x.split(' ')[0])

city_encoder=LabelEncoder()

shops['city_id']=city_encoder.fit_transform(shops['city'])

all_data['city_id']=all_data['shop_id'].map(shops.set_index('shop_id')['city_id'])

test['city_id']=test['shop_id'].map(shops.set_index('shop_id')['city_id'])
split=item_categories['item_category_name'].map(lambda x: x.split('-'))

item_categories['type_1']=split.map(lambda x:x[0].strip())

item_categories['type_2']=split.map(lambda x:x[1].strip() if len(x)>1 else x[0].strip())
## Label encoding type 1 of categories

item_encoder=LabelEncoder()

item_categories['type_1_encoded']=item_encoder.fit_transform(item_categories['type_1'])

## Label encoding type 2 of categories

item_encoder=LabelEncoder()

item_categories['type_2_encoded']=item_encoder.fit_transform(item_categories['type_2'])

## Map the encoded labels to all_data

all_data['item_category_type_1']=all_data['item_category_id'].map(item_categories.set_index('item_category_id')['type_1_encoded'])

all_data['item_category_type_2']=all_data['item_category_id'].map(item_categories.set_index('item_category_id')['type_2_encoded'])
all_data['city_id_cv_enc']=CV_encoder(all_data[['city_id','item_cnt_month']])

all_data['item_category_type_1_cv_enc']=CV_encoder(all_data[['item_category_type_1','item_cnt_month']])

all_data['item_category_type_2_cv_enc']=CV_encoder(all_data[['item_category_type_2','item_cnt_month']])
test['item_category_type_1']=test['item_category_id'].map(item_categories.set_index('item_category_id')['type_1_encoded'])

test['item_category_type_2']=test['item_category_id'].map(item_categories.set_index('item_category_id')['type_2_encoded'])



test['item_category_type_1_cv_enc']=test['item_category_type_1'].map(all_data.groupby('item_category_type_1')['item_cnt_month'].mean())

test['item_category_type_2_cv_enc']=test['item_category_type_2'].map(all_data.groupby('item_category_type_2')['item_cnt_month'].mean())



test['city_id_cv_enc']=test['city_id'].map(all_data.groupby('city_id')['item_cnt_month'].mean())
all_data.head()
test.head()
input_cols=['date_block_num','item_price','item_cnt_month','item_id_cv_enc','shop_id_cv_enc','item_category_id_cv_enc',

          'city_id_cv_enc','item_category_type_1_cv_enc','item_category_type_2_cv_enc']



predict_cols=all_data[input_cols].drop('item_cnt_month',axis=1).columns
lr=model_evaluator(all_data[input_cols])

test_preds=lr.predict(test[predict_cols])

sample_submission_copy=sample_submission

sample_submission_copy['item_cnt_month']=test_preds

sample_submission_copy.to_csv('lr_with_text.csv',index=False)

pd.read_csv('lr_with_text.csv')
all_data.to_pickle('all_data_numeric_text.pkl')

test.to_pickle('test_numeric_text.pkl')
all_data=pd.read_pickle('all_data_numeric_text.pkl')

test=pd.read_pickle('test_numeric_text.pkl')
## declaring input cols and predict cols

input_cols=['date_block_num','item_price','item_cnt_month','item_id_cv_enc','shop_id_cv_enc','item_category_id_cv_enc',

          'city_id_cv_enc','item_category_type_1_cv_enc','item_category_type_2_cv_enc']



predict_cols=all_data[input_cols].drop('item_cnt_month',axis=1).columns



##Linear model training

model=model_evaluator(all_data[input_cols])
##creating monthly lag features for sales: 1,2,3,6,12 months

all_data=pd.read_pickle('all_data_numeric_text.pkl')

test=pd.read_pickle('test_numeric_text.pkl')



lags=[1,2,3,6,12]

for i in lags:

    a=all_data[['date_block_num','shop_id','item_id','item_cnt_month']]

    a['date_block_num']=a['date_block_num']+i

    a.columns=['date_block_num','shop_id','item_id','item_cnt_month_lag_'+str(i)]

    all_data=pd.merge(all_data,a,on=['date_block_num','shop_id','item_id'],how='left')

    test=pd.merge(test,a,on=['date_block_num','shop_id','item_id'],how='left')



all_data.fillna(0,inplace=True)

test.fillna(0,inplace=True)

all_data.to_pickle('all_data_with_lag.pkl')
## evaluating model performance after including lag_data

model=model_evaluator(all_data)

test_preds=model.predict(test)
all_data=pd.read_pickle('all_data_with_lag.pkl')



from sklearn.feature_selection import SelectKBest, f_regression

selector=SelectKBest(f_regression,k=10)

train_set=all_data.loc[all_data['date_block_num']<33]

X_new=selector.fit_transform(train_set.drop('item_cnt_month',axis=1),train_set['item_cnt_month'])

selected_features=pd.DataFrame(data=selector.inverse_transform(X_new),columns=train_set.drop('item_cnt_month',axis=1).columns,index=train_set.index)

selected_columns=selected_features.columns[selected_features.var()!=0]



selected_columns_full_test=selected_columns.insert(0,'date_block_num')

selected_columns_full=selected_columns_full_test.insert(selected_columns.shape[0],'item_cnt_month')



print('selected_columns_full_test:',selected_columns_full_test)

print('selected_columns_full:',selected_columns_full)
##Evaluating feature scores

feature_scores=pd.DataFrame({'feature_name':all_data.columns.drop('item_cnt_month'),'feature_importance':selector.scores_})



##plotting the feature scores

sns.barplot(data=feature_scores,y='feature_name', x='feature_importance')
def rf_evaluator(ad,curr_block_num):

    depth=7

    model=RandomForestRegressor(max_depth=depth,n_estimators=50,n_jobs=-1,max_samples=0.8, max_features=0.8)

    X_train = ad.loc[ad['date_block_num']<curr_block_num].drop(['item_cnt_month'],axis=1)

    y_train = ad.loc[ad['date_block_num']<curr_block_num]['item_cnt_month']

    X_valid = ad.loc[ad['date_block_num']==curr_block_num].drop(['item_cnt_month'],axis=1)

    y_valid = ad.loc[ad['date_block_num']==curr_block_num]['item_cnt_month']

    model.fit(X_train,y_train)

    preds_valid=model.predict(X_valid)

    

    print('rmse for validation data',np.sqrt(mean_squared_error(y_valid,preds_valid)))

    print('r2 error',r2_score(y_valid,preds_valid))

    return model
def lr_evaluator(ad,curr_block_num):

    model=LinearRegression()

    X_train = ad.loc[ad['date_block_num']<curr_block_num].drop(['item_cnt_month'],axis=1)

    y_train = ad.loc[ad['date_block_num']<curr_block_num]['item_cnt_month']

    X_valid = ad.loc[ad['date_block_num']==curr_block_num].drop(['item_cnt_month'],axis=1)

    y_valid = ad.loc[ad['date_block_num']==curr_block_num]['item_cnt_month']

    model.fit(X_train,y_train)

    preds_valid=model.predict(X_valid)

    

    print('rmse for validation data',np.sqrt(mean_squared_error(y_valid,preds_valid)))

    print('r2 error',r2_score(y_valid,preds_valid))

    return model
block_lvl2=[29,30,31,32,33]

train_level2=np.zeros((all_data.loc[all_data['date_block_num'].isin(block_lvl2)].shape[0],2))

i=0



## training till a particular date and predicting outcome for the following date, hence creating meta predictions

for curr_date in block_lvl2:

    ts=time.time()

    ad=all_data.loc[all_data['date_block_num']<=curr_date]

    print('training linear model for date_block ',curr_date)

    lr=lr_evaluator(ad,curr_date)

    print('training random_forest model for date_block ',curr_date)

    rf=rf_evaluator(ad,curr_date)

    print('training done for curr_date ',curr_date)

    lr_preds=lr.predict(all_data.loc[all_data['date_block_num']==curr_date].drop('item_cnt_month',axis=1))

    rf_preds=rf.predict(all_data.loc[all_data['date_block_num']==curr_date].drop('item_cnt_month',axis=1))

    train_level2[i:i+lr_preds.shape[0],0]=lr_preds

    train_level2[i:i+lr_preds.shape[0],1]=rf_preds 

    i=i+lr_preds.shape[0]

    print('time taken for date_block ',curr_date,'is',-ts+time.time())
## pickiling the meta preds so as to not calculate it everytime

pd.DataFrame(train_level2).to_pickle('train_level2_rf_and_lr_full.pkl')

train_level2=pd.read_pickle('train_level2_rf_and_lr_full.pkl').values
## calculating meta data for test data

curr_date=33

ts=time.time()

ad=all_data.loc[all_data['date_block_num']<=curr_date]

print('training linear model for date_block ',curr_date)

lr=lr_evaluator(ad,curr_date)

lr_preds_test=lr.predict(test)

print('training random_forest model for date_block ',curr_date)

rf=rf_evaluator(ad,curr_date)

rf_preds_test=rf.predict(test)

print('training done for curr_date ',curr_date)
test_level2=np.c_[lr_preds_test,rf_preds_test]
target_level2=all_data.loc[all_data['date_block_num'].isin(block_lvl2)]['item_cnt_month']
lr2=LinearRegression().fit(train_level2,target_level2)



## calucating test predictions using the stacked model

test_preds=lr2.predict(test_level2)

sample_submission_copy=sample_submission

sample_submission_copy['item_cnt_month']=test_preds

sample_submission_copy.to_csv('stacked1.csv',index=False)

pd.read_csv('stacked1.csv')