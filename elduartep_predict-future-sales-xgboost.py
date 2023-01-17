train_xgb = False
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_all = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
sample = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
tes  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
from stop_words import get_stop_words
stop_words = get_stop_words('ru')
stop_words.append('ул')
stop_words.append('d')
import string
shops['shop_name'] = shops['shop_name'].str.replace('[{}]'.format(string.punctuation), '')
shops['shop_name'] = shops['shop_name'].str.lower()
shops["shop_name"] = shops["shop_name"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word.lower() not in stop_words))
shops['shop_name'] = shops['shop_name'].str.split()
shops["shop_name0"] = shops.shop_name.map( lambda x: x[0] )
shops["shop_name1"] = shops.shop_name.map( lambda x: x[1] )
shops["shop_name2"] = shops.shop_name.map( lambda x: x[2] if len(x)>2 else 'other')
shops.drop('shop_name',axis=1,inplace=True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df = pd.concat([shops.shop_name0,shops.shop_name1,shops.shop_name2],axis=0)
le.fit(list(df.unique()))
shops.shop_name0=le.transform(shops.shop_name0)
shops.shop_name1=le.transform(shops.shop_name1)
shops.shop_name2=le.transform(shops.shop_name2)
categories['item_category_name'] = categories['item_category_name'].str.replace('[{}]'.format(string.punctuation),' ')
categories['item_category_name'] = categories['item_category_name'].str.lower()
categories["item_category_name"] = categories["item_category_name"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word.lower() not in stop_words))
categories['item_category_name'] = categories['item_category_name'].str.split()
categories["item_category_name0"] = categories.item_category_name.map( lambda x: x[0] )
categories["item_category_name1"] = categories.item_category_name.map( lambda x: x[1] if len(x)>1 else 'other')
categories["item_category_name2"] = categories.item_category_name.map( lambda x: x[2] if len(x)>2 else 'other')
categories.drop('item_category_name',axis=1,inplace=True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df=pd.concat([categories.item_category_name0,categories.item_category_name1,categories.item_category_name2],axis=0)
le.fit(list(df.unique()))
categories.item_category_name0=le.transform(categories.item_category_name0)
categories.item_category_name1=le.transform(categories.item_category_name1)
categories.item_category_name2=le.transform(categories.item_category_name2)
items['item_name'] = items['item_name'].str.replace('[{}]'.format(string.punctuation), '')
items['item_name'] = items['item_name'].str.lower()
items["item_name"] = items["item_name"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word.lower() not in stop_words))
items['item_name'] = items['item_name'].str.split()
items["item_name0"] = items.item_name.map( lambda x: x[0] if len(x)>0 else 'other')
items["item_name1"] = items.item_name.map( lambda x: x[1] if len(x)>1 else 'other')
items["item_name2"] = items.item_name.map( lambda x: x[2] if len(x)>2 else 'other')
items.drop('item_name',axis=1,inplace=True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df=pd.concat([items.item_name0,items.item_name1,items.item_name2],axis=0)
le.fit(list(df.unique()))
items.item_name0=le.transform(items.item_name0)
items.item_name1=le.transform(items.item_name1)
items.item_name2=le.transform(items.item_name2)
del df
test['date_block_num'] = 34
test.drop('ID',axis=1,inplace=True)
train_all.drop('date',axis=1,inplace=True)
train_all = train_all[(train_all.item_price < 300000 )& (train_all.item_cnt_day < 1000)]
train_all = pd.merge(train_all,items[['item_id','item_category_id']], on='item_id', how='left')
test      = pd.merge(test     ,items[['item_id','item_category_id']], on='item_id', how='left')
#assigning price to test items

# average category price
means = train_all[train_all.date_block_num<34].groupby('item_category_id').item_price.mean()
train_all['cat_price'] = train_all['item_category_id'].map(means)
test['cat_price'] = test['item_category_id'].map(means)

#for test data: item price = average category price
test['item_price'] = test['cat_price']

#average item price
means = train_all[train_all.date_block_num<34].groupby('item_id').item_price.mean()
train_all['item_avg_price'] = train_all['item_id'].map(means)
test['item_avg_price'] = test['item_id'].map(means)
test['item_avg_price'] = test.apply(lambda x: x.cat_price if np.isnan(x.item_avg_price) else x.item_avg_price,axis=1)


##averagecategory shop price
cols = ['item_category_id','shop_id']
means = train_all[train_all.date_block_num<34].groupby( cols ).agg({"item_price":["mean"]})
means.columns = ["cat_shop_price"]
means.reset_index( inplace = True)
test = pd.merge( test, means, on = cols, how = "left" )
train_all = pd.merge( train_all, means, on = cols, how = "left" )
#assingning price to test item_cat that is not present in training
means = train_all[train_all.date_block_num<34].groupby('item_category_id').cat_shop_price.mean()
test['tmp'] = test['item_category_id'].map(means)
test['cat_shop_price'] = test.apply(lambda x: x.tmp if np.isnan(x.cat_shop_price) else x.cat_price,axis=1)


##average item shop price
cols = ['item_id','shop_id']
means = train_all[train_all.date_block_num<34].groupby( cols ).agg({"item_price":["mean"]})
means.columns = ["item_shop_price"]
means.reset_index( inplace = True)
test = pd.merge( test, means, on = cols, how = "left" )
train_all = pd.merge( train_all, means, on = cols, how = "left" )
#assingning price to test item_cat that is not present in training
#means = train_all[train_all.date_block_num<34].groupby('item_category_id').item_shop_price.mean()
#test['tmp'] = test['item_category_id'].map(means)
test['item_shop_price'] = test.apply(lambda x: x.cat_shop_price if np.isnan(x.item_shop_price) else x.item_shop_price,axis=1)
train_all = pd.concat([train_all, test], ignore_index=True)
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='uniform')

mi = train_all['item_avg_price'].min()
train_all['item_avg_price'] = np.log(train_all['item_avg_price']-mi+1).astype(np.float16)
est.fit(train_all['item_avg_price'].values.reshape(-1,1))
train_all['BIN_item_price'] = pd.DataFrame(est.transform(train_all['item_avg_price'].values.reshape(-1,1)), columns=['BIN_item_price']).astype(np.int8)#.astype(str)

mi = train_all['cat_price'].min()
train_all['cat_price'] = np.log(train_all['cat_price']-mi+1).astype(np.float16)
est.fit(train_all['cat_price'].values.reshape(-1,1))
train_all['BIN_cat_price'] = pd.DataFrame(est.transform(train_all['cat_price'].values.reshape(-1,1)), columns=['BIN_cat_price']).astype(np.int8)#.astype(str)

mi = train_all['cat_shop_price'].min()
train_all['cat_shop_price'] = np.log(train_all['cat_shop_price']-mi+1).astype(np.float16)
est.fit(train_all['cat_shop_price'].values.reshape(-1,1))
train_all['BIN_cat_shop_price'] = pd.DataFrame(est.transform(train_all['cat_shop_price'].values.reshape(-1,1)), columns=['BIN_cat_shop_price']).astype(np.int8)#.astype(str)

mi = train_all['item_shop_price'].min()
train_all['item_shop_price'] = np.log(train_all['item_shop_price']-mi+1).astype(np.float16)
est.fit(train_all['item_shop_price'].values.reshape(-1,1))
train_all['BIN_item_shop_price'] = pd.DataFrame(est.transform(train_all['item_shop_price'].values.reshape(-1,1)), columns=['BIN_item_shop_price']).astype(np.int8)#.astype(str)

plt.hist(train_all.item_avg_price,bins=50);
train_all['revenue'] = (train_all['item_cnt_day'] * train_all['item_price']).astype(np.float32)

train_all['target']  = train_all['date_block_num'].astype(str)+'_'+train_all['shop_id'].astype(str)+'_'+train_all['item_id'].astype(str)
means                = train_all.groupby(['target'])['item_cnt_day'].sum()
train_all['target']  = train_all['target'].map(means)
train_all.target.clip(0, 20,inplace=True)
train_all[train_all.date_block_num<34].groupby('date_block_num').target.mean().mean()
##############################################################
### target, revenue and item_price oversampling item x shop
##############################################################

train_mes=[]
from itertools import product
for i in range(34):
    one_month = train_all[train_all.date_block_num == i]
    train_mes.append( np.array(list( product( [i], one_month.shop_id.unique(), one_month.item_id.unique() ) ), dtype = np.int16) )
del one_month

cols  = ["date_block_num", "shop_id", "item_id"]
train_mes = pd.DataFrame( np.vstack(train_mes), columns = cols )

means = train_all.groupby( cols ).agg( {"item_cnt_day": ["sum"]} )
means.columns = ["target"]
means.reset_index( inplace = True)
train_mes = pd.merge( train_mes, means, on = cols, how = "left" )

means = train_all.groupby( cols ).agg( {"revenue": ["mean"]} )
means.columns = ["revenue"]
means.reset_index( inplace = True)
train_mes = pd.merge( train_mes, means, on = cols, how = "left" )

means = train_all.groupby( cols ).agg( {"item_price": ["mean"]} )
means.columns = ["item_price"]
means.reset_index( inplace = True)
train_mes = pd.merge( train_mes, means, on = cols, how = "left" )

train_mes = pd.concat([train_mes, train_all[train_all['date_block_num']==34][cols]], ignore_index=True)
train_mes.fillna(0,inplace=True)
train_mes.target.clip(0, 20,inplace=True)
train_mes[train_mes.date_block_num<34].groupby('date_block_num').target.mean().mean()
train_mes = pd.merge(train_mes,items, on='item_id', how='left')
train_mes = pd.merge(train_mes,categories, on='item_category_id', how='left')
train_mes = pd.merge(train_mes,shops, on='shop_id', how='left')
means = train_all.groupby('item_id').BIN_item_price.mean()
train_mes['BIN_item_price'] = train_mes['item_id'].map(means)#.astype(str)

means = train_all.groupby('item_category_id').BIN_cat_price.mean()
train_mes['BIN_cat_price'] = train_mes['item_category_id'].map(means)#.astype(str)

cols  = ["shop_id", "item_category_id"]
means = train_all.groupby( cols ).agg( {"BIN_cat_shop_price": ["sum"]} )
means.columns = ["BIN_cat_shop_price"]
means.reset_index( inplace = True)
train_mes = pd.merge( train_mes, means, on = cols, how = "left" )
#there are a lot of shop_id x other_variables that there are in train_mes that there are not in train
train_mes['BIN_cat_shop_price'].fillna(int(train_mes['BIN_cat_shop_price'].mode().values[0]),inplace=True)

cols  = ["shop_id", "item_id"]
means = train_all.groupby( cols ).agg( {"BIN_item_shop_price": ["sum"]} )
means.columns = ["BIN_item_shop_price"]
means.reset_index( inplace = True)
train_mes = pd.merge( train_mes, means, on = cols, how = "left" )
#there are a lot of shop_id x other_variables that there are in train_mes that there are not in train
train_mes['BIN_item_shop_price'].fillna(train_mes['BIN_item_shop_price'].mode().values[0],inplace=True)
plt.xlabel('month')
plt.ylabel('total_item_cnt_month')
t = np.linspace(0, 35, 35*5)
plt.plot(t, 0.285+0.03*np.cos(2 * np.pi  * (t+1) /12))
plt.scatter(np.arange(34),train_mes[train_mes.date_block_num<34].groupby('date_block_num').target.mean())
d = {'date_block_num': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34],
     'tot_days'      : [31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30],
     'L1_days'       : [30,31,28,31,30,31,30,31,31,30,31,30,30,31,28,31,30,31,30,31,31,30,31,30,30,31,28,31,30,31,30,31,31,30,31],
     'L2_days'       : [30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30],
     'L3_days'       : [31,30,28,31,28,31,30,31,30,31,31,30,31,30,28,31,28,31,30,31,30,31,31,30,31,30,28,31,28,31,30,31,30,31,31],
    }
days = pd.DataFrame(data=d)

days['dec'] = (((days.date_block_num+1) % 12)==0).astype(np.int8)
days['month'] = ((days.date_block_num+1) % 12)

#lag date_block_num
days['L1'] = days['date_block_num'].apply(lambda x: x-2 if (x%12)==0 else x-1 )
days['L2'] = days['date_block_num'].apply(lambda x: x-1 if (x%12)==1 else x-2 )
days['L3'] = days['date_block_num'].apply(lambda x: x-1 if (x%12)==2 else x-3 )

days['tot_days']    = days['tot_days'].astype(np.int8)
#lag number of days for each month
days['L1_days']     = (days['tot_days']/days['L1_days']).astype(np.float16)
days['L2_days']     = (days['tot_days']/days['L2_days']).astype(np.float16)
days['L3_days']     = (days['tot_days']/days['L3_days']).astype(np.float16)

#seasonal behaviour
days['sin_'] = 0.285+0.03*(1+4*days['dec'])*np.cos(2 * np.pi  * (days['date_block_num']+1) /12)
days['sin'] = np.cos(2 * np.pi  * (days['date_block_num']+1) /12)
                                                                         
train_mes = pd.merge(train_mes,days, on='date_block_num', how='left')

del d
del days
train_mes["item_shop_first_sale"] = (train_mes["date_block_num"] - train_mes.groupby(["item_id","shop_id"])["date_block_num"].transform('min')).astype(np.int8)
train_mes["cat_shop_first_sale"]  = (train_mes["date_block_num"] - train_mes.groupby(["item_category_id","shop_id"])["date_block_num"].transform('min')).astype(np.int8)
train_mes["item_first_sale"]      = (train_mes["date_block_num"] - train_mes.groupby(["item_id"])["date_block_num"].transform('min')).astype(np.int8)
train_mes["cat_first_sale"]       = (train_mes["date_block_num"] - train_mes.groupby(["item_category_id"])["date_block_num"].transform('min')).astype(np.int8)
tmp = np.zeros((3,len(train_mes)))

def lag_features(train_mes,cols_to_group=['date_block_num','item_id','shop_id'],aggregate_on='target',name='month_itemN_target',lags=3):
    
    L = ['L1'+name,'L2'+name,'L3'+name]
    Ld = ['L1_days','L3_days','L3_days']
    D = 'D'+name
    m = ['L1','L2','L3']

    cols = ['L1','L2','L3']
    cols = cols + cols_to_group
    cols.append(aggregate_on)
    cols_to_merge = cols_to_group.copy()

    means = train_mes.groupby(cols_to_group).agg({aggregate_on:['mean']})
    means.columns = [name]
    means.reset_index(inplace = True)
    df = train_mes[cols]
    df = pd.merge(df,means,on=cols_to_group,how='left')
    for lag in [0,1,2]:
        dfc = df.copy()
        cols = list(dfc.columns)
        cols[lag], cols[3] = cols[3], cols[lag]
        cols[-1] = L[lag]
        dfc.columns = cols
        cols_to_merge[0] = m[lag]
        cols_ = [L[lag]]
        cols_ = cols_ + cols_to_merge
        dfc.reset_index(inplace = True)
        dfc = dfc[cols_]
        dfc.drop_duplicates(inplace=True)
        train_mes = pd.merge(train_mes, dfc, on=cols_to_merge,how="left")
        train_mes[L[lag]] = (train_mes[L[lag]]*train_mes[Ld[lag]]).astype(np.float16)
    tmp[0] = (2*train_mes[L[0]]-train_mes[L[1]])    
    tmp[1] = (3*train_mes[L[1]]-2*train_mes[L[2]])
    tmp[2] = (3*train_mes[L[0]]-train_mes[L[2]])*0.5
    train_mes[D] = pd.Series(np.mean(tmp,axis=0)).astype(np.float16)
    train_mes[D].fillna(0,inplace=True)

    if lags >1:
        tmp[0] = train_mes[L[0]]
        tmp[1] = train_mes[L[1]]
    if lags >2:
        tmp[2] = train_mes[L[2]]

    for lag in [0,1,2]:
        if lag<lags:
            if lag == 1 :
                train_mes[L[1]] = pd.Series(np.mean(tmp[:2,:],axis=0)).astype(np.float16)
            if lag == 2 :
                train_mes[L[2]] = pd.Series(np.mean(tmp,axis=0)).astype(np.float16)
            train_mes[L[lag]].fillna(0,inplace=True)
        else:
            train_mes.drop(L[lag],axis=1,inplace=True)   
    return train_mes
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_id','shop_id'],aggregate_on='target',name='_target',lags=1)
plt.xlabel('month')
plt.ylabel('normalized previous_value_benchmark')
plt.scatter(train_mes[train_mes.date_block_num<34]['date_block_num'].sample(200,random_state=0), 0.285+0.03*(1+4*train_mes[train_mes.date_block_num<34]['dec'].sample(200,random_state=0))*np.cos(2 * np.pi  * (train_mes[train_mes.date_block_num<34]['date_block_num'].sample(200,random_state=0)+1) /12))
plt.scatter(np.arange(34),train_mes[train_mes.date_block_num<34].groupby('date_block_num').L1_target.mean())
tes  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
predictions = tes
predictions['item_cnt_month'] = train_mes[train_mes.date_block_num==34].L1_target.values
predictions = predictions.drop(['shop_id','item_id'],axis=1)
predictions.to_csv('out_L1.csv',index=False)

predictions.item_cnt_month.mean()
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_id'],aggregate_on='target',name='month_item_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','shop_id'],aggregate_on='target',name='month_shop_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_category_id'],aggregate_on='target',name='month_cat_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_category_id','shop_id'],aggregate_on='target',name='month_cat_shop_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_category_id','item_name1',     ],aggregate_on='target',name='month_cat_itemM_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_category_id','item_name2',     ],aggregate_on='target',name='month_cat_itemN_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_category_id','BIN_item_price'  ],aggregate_on='target',name='month_cat_itemB_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','BIN_cat_price','shop_id'],aggregate_on='target',name='month_shop_itemC_target',lags=1)

train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_category_id','BIN_item_shop_price'],aggregate_on='target',name='month_cat_itemS_target',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','BIN_cat_price'],aggregate_on='target',name='month_itemC_target',lags=1)

train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_name1'],aggregate_on='revenue',name='month_itemM_revenue',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_name2'],aggregate_on='revenue',name='month_itemN_revenue',lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','item_id'],   aggregate_on='revenue',name='month_item_revenue', lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','shop_id'],   aggregate_on='revenue',name='month_shop_revenue', lags=1)
train_mes=lag_features(train_mes,cols_to_group=['date_block_num','BIN_cat_price'],aggregate_on='revenue',name='month_itemC_revenue',lags=1)
del tmp
del test
del train_all
del means
train_mes["date_block_num"]      = train_mes["date_block_num"].astype(np.int8)
train_mes['month']               = train_mes['month'].astype(np.int8)
train_mes["item_category_id"]    = train_mes["item_category_id"].astype(np.float16).astype(np.int8)
train_mes["shop_id"]             = train_mes["shop_id"].astype(np.int8)
train_mes["item_id"]             = train_mes["item_id"].astype(np.int16)

train_mes["BIN_cat_shop_price"]  = train_mes["BIN_cat_shop_price"].astype(np.int16)
train_mes["BIN_item_shop_price"] = train_mes["BIN_item_shop_price"].astype(np.int8)

train_mes["item_name0"]          = train_mes["item_name0"].astype(np.int16)
train_mes["item_name1"]          = train_mes["item_name1"].astype(np.int16)
train_mes["item_name2"]          = train_mes["item_name2"].astype(np.int16)
train_mes["item_category_name0"] = train_mes["item_category_name0"].astype(np.int16)
train_mes["item_category_name1"] = train_mes["item_category_name1"].astype(np.int16)
train_mes["item_category_name2"] = train_mes["item_category_name2"].astype(np.int16)
train_mes["shop_name0"]          = train_mes["shop_name0"].astype(np.int16)
train_mes["shop_name1"]          = train_mes["shop_name1"].astype(np.int16)
train_mes["shop_name2"]          = train_mes["shop_name2"].astype(np.int16)

train_mes["sin_"]                = train_mes["sin_"].astype(np.float16)
train_mes.replace([np.inf, -np.inf], np.nan,inplace=True)
train_mes.fillna(0,inplace=True)
print(train_mes.isnull().values.any())
print(train_mes.isnull().sum().sum())
train_mes.drop(['revenue','item_price','L1_days', 'L2_days', 'L3_days',
                'shop_name0','shop_name1','item_category_name0',
                'BIN_item_price',
                'BIN_item_shop_price',
                'BIN_cat_price',
                'sin',
                'L1','L2','L3'],axis=1,inplace=True)
dummy = train_mes.loc[:1]
dummy.drop('target',axis=1,inplace=True)
train_cols = dummy.columns
del dummy
len(train_cols)
for col in train_mes.columns:
    print(col,type(train_mes[col][0]))
from sklearn.utils import shuffle

X_train = train_mes[ (train_mes['date_block_num'] <  33) & (train_mes['date_block_num'] >2) ]
X_train = shuffle(X_train)
y_train = X_train['target']
X_train = X_train[train_cols]
X_valid = train_mes[ train_mes['date_block_num'] == 33 ][train_cols]
y_valid = train_mes[ train_mes['date_block_num'] == 33 ]['target']
X_test  = train_mes[ train_mes['date_block_num'] == 34 ][train_cols]

train_len = len(X_train)
X_train_ensamble =X_train_ensamble[X_train_ensamble.date_block_num>2]
del train_mes
import gc

gc.collect();
from xgboost import XGBRegressor

if train_xgb:
    xgb = XGBRegressor(
                    max_depth        = 10,
                    n_estimators     = 1000,
                    seed             = 3562,
                    learning_rate    = 0.1, 
                    min_child_weight = 5,       #~min number of items in a node to further splitting
                    colsample_bytree = 0.7,     #features
                    subsample        = 0.7,     #items
                    gamma            = 0.0005,   #Minimum loss reduction required to make a further partition
                    reg_alpha        = 3.5,       #l1 regulatization
                    reg_lambda       = 3.5        #l2 regularization
    )
if train_xgb:
    xgb.fit(X_train,
            y_train,
            eval_metric="rmse",
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=10)
import joblib

if train_xgb:
    #save model
    joblib.dump(xgb, 'xgboost_model')
else:
    #load saved model
    xgb = joblib.load('../input/predict-future-saless/xgboost_model')
from xgboost import plot_importance

def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(xgb, (10,24))
xgboost_pred  = xgb.predict(X_test).clip(0, 20)

predictions  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
predictions['item_cnt_month'] = xgboost_pred
predictions = predictions.drop(['shop_id','item_id'],axis=1)
predictions.to_csv('out_xgb.csv',index=False)

predictions.item_cnt_month.mean()