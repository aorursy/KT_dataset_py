import numpy as np

import pandas as pd 



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(palette=sns.color_palette('Set2',9))



from sklearn.preprocessing import LabelEncoder



from xgboost import XGBRegressor
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path='/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv'

test_path='/kaggle/input/competitive-data-science-predict-future-sales/test.csv'

train=pd.read_csv(train_path)

test=pd.read_csv(test_path)



items_path='/kaggle/input/competitive-data-science-predict-future-sales/items.csv'

item_categories_path='/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv'

shops_path='/kaggle/input/competitive-data-science-predict-future-sales/shops.csv'

items=pd.read_csv(items_path)

item_cat=pd.read_csv(item_categories_path)

shops=pd.read_csv(shops_path)



sample_submission_path='/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv'

sample_submission=pd.read_csv(sample_submission_path)
def printinfo(df):

    print('\n\n********info*********\n\n',df.info(),'\n\n********head*********\n\n',df.head(),'\n\n********describe*********\n\n',df.describe())
printinfo(train)
printinfo(test)
f1,ax1=plt.subplots(1,1,figsize=(10,2))

sns.boxplot(x=train.item_cnt_day,ax=ax1)

f2,ax2=plt.subplots(1,1,figsize=(10,2))

sns.boxplot(x=train.item_price,ax=ax2)
train=train[train.item_price<100000]

train=train[train.item_cnt_day<1100]
train[train.item_price<0]
median=train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].median()

train.loc[train.item_price<0,'item_price']=median
#duplicated shop id

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57



train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58



train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
train[train.item_price.isnull()]
train[(train.item_id==2973)&(train.shop_id==32)]
train.loc[train.item_price.isnull(),'item_price']=1249.0
shops.loc[shops.shop_name=='Сергиев Посад ТЦ "7Я"','shop_name']='СергиевПосад ТЦ "7Я"'

shops['city']=shops['shop_name'].str.split(' ').map(lambda x:x[0])

shops['city_code']=LabelEncoder().fit_transform(shops['city'])

shops=shops[['shop_id','city_code']]
item_cat['split']=item_cat['item_category_name'].str.split('-')

item_cat['type']=item_cat['split'].map(lambda x:x[0].strip())

item_cat['type_code']=LabelEncoder().fit_transform(item_cat['type'])



def subtype(x):                         

    if len(x)>2:                  #exception for Blu-Ray

        return x[1].strip()+x[2].strip()

    elif len(x)>1:

        return x[1].strip()

    else:

        return x[0].strip()

item_cat['subtype']=item_cat['split'].map(subtype)

item_cat['subtype_code']=LabelEncoder().fit_transform(item_cat['subtype'])

item_cat=item_cat[['item_category_id','type_code','subtype_code']]
items.drop(['item_name'],axis=1,inplace=True)
print('{} shops,{} items in {} test set rows '.format(test.shop_id.unique().shape[0],test.item_id.unique().shape[0],test.shape[0]))
new_items=len(set(test.item_id)-set(test.item_id).intersection(set(train.item_id)))

print('There are {} new items in test set'.format(new_items))
from itertools import product

matrix=[]

cols=['date_block_num','shop_id','item_id']

for i in range(34):

    sales=train[train.date_block_num==i]

    matrix.append(np.array(list(product([i],sales.shop_id.unique(),sales.item_id.unique()))))
matrix=pd.DataFrame(np.vstack(matrix),columns=cols)

matrix.sort_values(cols,inplace=True)
train['revenue']=train['item_price']*train['item_cnt_day']
group=train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':'sum'})

group.columns=['item_cnt_month']

group.reset_index(inplace=True)
matrix=pd.merge(matrix,group,on=cols,how='left')

matrix['item_cnt_month']=matrix['item_cnt_month'].fillna(0).clip(0,20)
test['date_block_num']=34

test.drop(['ID'],axis=1,inplace=True)

matrix=pd.concat([matrix,test],ignore_index=True)

matrix['item_cnt_month'].fillna(0,inplace=True)
matrix=pd.merge(matrix,shops,on='shop_id',how='left')   #add city code

matrix=pd.merge(matrix,items,on='item_id',how='left')   #add item cateogory
matrix=pd.merge(matrix,item_cat,on='item_category_id',how='left') #add type and subtype
matrix[['date_block_num','shop_id','city_code',

       'item_category_id','type_code','subtype_code']]=matrix[['date_block_num','shop_id','city_code',

                                                             'item_category_id','type_code','subtype_code']].astype(np.int8)

matrix[['item_id']]=matrix[['item_id']].astype(np.int16)

matrix[['item_cnt_month']]=matrix[['item_cnt_month']].astype(np.float16)
matrix.info()
def lag_feature(df,lags,col):

    tmp=df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted=tmp.copy()

        shifted.columns=['date_block_num','shop_id','item_id',col+'_lag'+str(i)]

        shifted['date_block_num']+=i

        df=pd.merge(df,shifted,on=['date_block_num','shop_id','item_id'],how='left')

    return df
matrix=lag_feature(matrix,[1,2,3,6,12],'item_cnt_month')  #add time lag feature
def add_mean_feature(df,addname,grouplist,time):   # df / ['addname']/ ['list1','list2',...] /[1,2,...]



    group_tmp=df.groupby(grouplist).agg({'item_cnt_month':'mean'})

    group_tmp.columns=addname

    group_tmp.reset_index(inplace=True)

    

    df=pd.merge(df,group_tmp,on=grouplist,how='left')

    df=lag_feature(df,time,addname[0])

    df.drop(addname,axis=1,inplace=True)

    

    return df
matrix=add_mean_feature(matrix,['date_avg_item_cnt'],['date_block_num'],[1])

matrix=add_mean_feature(matrix,['date_item_avg_item_cnt'],['date_block_num','item_id'],[1,2,3,6,12])

matrix=add_mean_feature(matrix,['date_shop_avg_item_cnt'],['date_block_num','shop_id'],[1,2,3,6,12])



matrix=add_mean_feature(matrix,['date_cat_avg_item_cnt'],['date_block_num','item_category_id'],[1])

matrix=add_mean_feature(matrix,['date_type_avg_item_cnt'],['date_block_num','type_code'],[1])

matrix=add_mean_feature(matrix,['date_subtype_avg_item_cnt'],['date_block_num','subtype_code'],[1])

matrix=add_mean_feature(matrix,['date_city_avg_item_cnt'],['date_block_num','city_code'],[1])



matrix=add_mean_feature(matrix,['date_shop_cat_avg_item_cnt'],['date_block_num','shop_id','item_category_id'],[1])

matrix=add_mean_feature(matrix,['date_shop_type_avg_item_cnt'],['date_block_num','shop_id','type_code'],[1])

matrix=add_mean_feature(matrix,['date_shop_subtype_avg_item_cnt'],['date_block_num','shop_id','subtype_code'],[1])



matrix=add_mean_feature(matrix,['date_item_city_avg_item_cnt'],['date_block_num','item_id','city_code'],[1])
train_f=train.drop(['date','item_cnt_day'],axis=1)

train_f[['date_block_num','shop_id']]=train_f[['date_block_num','shop_id']].astype(np.int8)

train_f[['item_id']]=train_f[['item_id']].astype(np.int16)

train_f[['item_price','revenue']]=train_f[['item_price','revenue']].astype(np.float16)

train_f.info()
def add_price_feature(mat,df,addname,grouplist):   # mat/ df / ['addname']/ ['list1','list2',...]



    group_tmp=df.groupby(grouplist).agg({'item_price':'mean'})

    group_tmp.columns=addname

    group_tmp[addname]=group_tmp[addname].astype(np.float16)

    group_tmp.reset_index(inplace=True)

    

    mat=pd.merge(mat,group_tmp,on=grouplist,how='left')

    

    return mat
matrix=add_price_feature(matrix,train_f,['item_avg_item_price'],['item_id'])

matrix=add_price_feature(matrix,train_f,['date_item_avg_item_price'],['date_block_num','item_id'])



lags=[1,2,3,4,5,6]

matrix=lag_feature(matrix,lags,'date_item_avg_item_price')

for i in lags:

    matrix['delta_price_lag'+str(i)]=(matrix['date_item_avg_item_price_lag'+str(i)]-matrix['item_avg_item_price'])/matrix['item_avg_item_price']
def select_trend(row):

    for i in lags:

        if row['delta_price_lag'+str(i)]:

            return row['delta_price_lag'+str(i)]

    return 0



matrix['delta_price_lag']=matrix.apply(select_trend,axis=1)

matrix['delta_price_lag'].fillna(0, inplace=True)

matrix['delta_price_lag']=matrix['delta_price_lag'].astype(np.float16)



drop_list=['item_avg_item_price','date_item_avg_item_price']

for i in lags:

    drop_list.append('date_item_avg_item_price_lag'+str(i))

    drop_list.append('delta_price_lag'+str(i))

matrix.drop(drop_list,axis=1,inplace=True)
def fun_first_item_shop(x):

    d={}

    d['first_sale_item_shop']=x.loc[x.item_cnt_month>0,'date_block_num'].min()

    return pd.Series(d)

def fun_first_item(x):

    d={}

    d['first_sale_item']=x.loc[x.item_cnt_month>0,'date_block_num'].min()

    return pd.Series(d)



def first(mat):

    group1=mat.groupby(['item_id','shop_id'])[['date_block_num','item_cnt_month','item_id','shop_id']].apply(fun_first_item_shop)

    group2=mat.groupby(['item_id'])[['date_block_num','item_cnt_month','item_id','shop_id']].apply(fun_first_item)

   

    mat=pd.merge(mat,group1,on=['item_id','shop_id'],how='left')

    mat=pd.merge(mat,group2,on=['item_id'],how='left')



    mat['first_sale_item_shop_period']=mat['date_block_num']-mat['first_sale_item_shop']

    mat['first_sale_item_period']=mat['date_block_num']-mat['first_sale_item']

   

    mat.loc[mat.first_sale_item_shop_period<0,'first_sale_item_shop_period']=0

    mat.loc[mat.first_sale_item_period<0,'first_sale_item_period']=0

    

    mat.drop(['first_sale_item_shop','first_sale_item'],axis=1,inplace=True)

    return mat
def fun_last_item_shop(x):

    d={}

    d['last_sale_item_shop']=x.loc[x.item_cnt_month>0,'date_block_num'].max()

    return pd.Series(d)

def fun_last_item(x):

    d={}

    d['last_sale_item']=x.loc[x.item_cnt_month>0,'date_block_num'].max()

    return pd.Series(d)



def last(mat):

    group1=mat.groupby(['item_id','shop_id'])[['date_block_num','item_cnt_month','item_id','shop_id']].apply(fun_last_item_shop)

    group2=mat.groupby(['item_id'])[['date_block_num','item_cnt_month','item_id','shop_id']].apply(fun_last_item)

    

    mat=pd.merge(mat,group1,on=['item_id','shop_id'],how='left')

    mat=pd.merge(mat,group2,on=['item_id'],how='left')

    

    mat['last_sale_item_shop_period']=mat['date_block_num']-mat['last_sale_item_shop']

    mat['last_sale_item_period']=mat['date_block_num']-mat['last_sale_item']

    

    mat.loc[mat.last_sale_item_shop_period<0,'last_sale_item_shop_period']=0

    mat.loc[mat.last_sale_item_period<0,'last_sale_item_period']=0

    

    mat.drop(['last_sale_item_shop','last_sale_item'],axis=1,inplace=True)

    return mat
matrix=first(matrix)

matrix=last(matrix)
frls_list=['first_sale_item_shop_period','first_sale_item_period','last_sale_item_shop_period','last_sale_item_period']
matrix[frls_list]=matrix[frls_list].fillna(0)
matrix[frls_list]=matrix[frls_list].astype(np.int8)
matrix['month'] = matrix['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])

matrix['days'] = matrix['month'].map(days).astype(np.int8)
matrix.to_pickle('data.pkl')

data = pd.read_pickle('data.pkl')
data = data[data.date_block_num > 11]
for col in data.columns:

    if '_lag' in col:

        data[col].fillna(0,inplace=True)
data.info()
data=data[[

 'date_block_num',

 'shop_id',

 'item_id',

 'item_cnt_month',

 'city_code',

 'item_category_id',

 'type_code',

 'subtype_code',

 'item_cnt_month_lag1',

 'item_cnt_month_lag2',

 'item_cnt_month_lag3',

 'item_cnt_month_lag6',

 'item_cnt_month_lag12',

 'date_avg_item_cnt_lag1',

 'date_item_avg_item_cnt_lag1',

 'date_item_avg_item_cnt_lag2',

 'date_item_avg_item_cnt_lag3',

 'date_item_avg_item_cnt_lag6',

 'date_item_avg_item_cnt_lag12',

 'date_shop_avg_item_cnt_lag1',

 'date_shop_avg_item_cnt_lag2',

 'date_shop_avg_item_cnt_lag3',

 'date_shop_avg_item_cnt_lag6',

 'date_shop_avg_item_cnt_lag12',

 'date_cat_avg_item_cnt_lag1',

 #'date_type_avg_item_cnt_lag1',

 #'date_subtype_avg_item_cnt_lag1',

 'date_city_avg_item_cnt_lag1',

 'date_shop_cat_avg_item_cnt_lag1',

 #'date_shop_type_avg_item_cnt_lag1',

 #'date_shop_subtype_avg_item_cnt_lag1',

 'date_item_city_avg_item_cnt_lag1',

 'delta_price_lag',

 'first_sale_item_shop_period',

 'first_sale_item_period',

 #'last_sale_item_shop_period',

 #'last_sale_item_period',

 'month'

]]
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = data[data.date_block_num < 33]['item_cnt_month']

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = data[data.date_block_num == 33]['item_cnt_month']

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
from xgboost import XGBRegressor



model = XGBRegressor(

    tree_method='gpu_hist',

    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    random_state=42)
model.fit(

    X_train, 

    Y_train, 

    

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    eval_metric="rmse",

    verbose=True, 

    early_stopping_rounds = 10)
Y_pred = model.predict(X_valid).clip(0, 20)

Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)
from xgboost import plot_importance

fig, ax = plt.subplots(1,1,figsize=(10,10))

plot_importance(booster=model, ax=ax)