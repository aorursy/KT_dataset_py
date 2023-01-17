import numpy as np 
import pandas as pd 
import datetime


train=pd.read_csv('../input/sales_train.csv')
print(train.head(10))

train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

train = train.drop(['date'], axis=1)

print(train.head(10))

#train.to_csv('train data separate.csv',index=False)
import matplotlib.pyplot as plt

print('原列数：')
print(train['item_price'].shape[0])

print('统计：')
print(train['item_price'].describe())

print('缺失数：')
print(train['item_price'].isnull().describe)

plt.hist(train['item_price'], bins=15)
plt.title('hist picture')#设置标题头
plt.xlabel('item price ')#x轴
plt.ylabel('item price distribution count')#y轴
plt.show()

plt.hist(train['item_price'], bins=15)
plt.ylim(0, 100)
plt.title('hist picture ylimit 100')
plt.xlabel('item price ')
plt.ylabel('item price distribution count')
plt.show()
print('train.item_price.min %d '% train.item_price.min())
print('train.item_price.max %d '% train.item_price.max())
print('train.item_price.mean %d '% train.item_price.mean())
print('train.item_price.median %d '% train.item_price.median())

print('------------------------------------------------')
print('inital train shape:', train.shape)
train = train[(train.item_price > 0) & (train.item_price < 300000)]
print('after choose train shape:', train.shape)

print('after choose train------------------------------')
print('train.item_price.min %d '% train.item_price.min())
print('train.item_price.max %d '% train.item_price.max())
print('train.item_price.mean %d '% train.item_price.mean())
print('train.item_price.median %d '% train.item_price.median())

print('------------------------------------------------')
print(train.item_price.value_counts().sort_index(ascending=False))

print('train.item_cnt_day.min: %d' % train.item_cnt_day.min())
print('train.item_cnt_day.max: %d' % train.item_cnt_day.max())
print('train.item_cnt_day.mean: %d' % train.item_cnt_day.mean())
print('train.item_cnt_day.median: %d' % train.item_cnt_day.median())

plt.hist(train['item_cnt_day'], bins=10)
plt.title('hist item cnt day')
plt.show()

train.groupby('date_block_num').sum()['item_cnt_day'].hist()
plt.title('Sales per month histogram')
plt.show()

plt.plot(train.groupby('date_block_num').sum()['item_cnt_day'])
plt.title('Sales per month')
plt.show()

#下面画图是新增加的
p=train[train['date_block_num']<12]
p1=train[train['date_block_num']>11]
p1=p1[p1['date_block_num']<24]
p2=train[train['date_block_num']>23]
print(p2.head(5))
plt.plot(p.groupby('month').sum()['item_cnt_day'])
plt.plot(p1.groupby('month').sum()['item_cnt_day'])
plt.plot(p2.groupby('month').sum()['item_cnt_day'])
plt.title('Three pictrue compare')
plt.show()

items = pd.read_csv('../input/items.csv')
train = train.drop(['item_price'], axis=1)

train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})

train = pd.merge(train,items, how='left', on=['item_id'])
train=train.drop(['item_name'], axis=1)

print(train.head(10))
print(train.isnull().any().describe())
print(train.isnull().all().describe())

plt.hist(train['item_cnt_month'], bins=15)
plt.title('item_cnt_month original')
plt.show()

y2= np.log1p(train['item_cnt_month'].clip(0.,20.))
plt.hist(y2,bins=20)
plt.title('item_cnt_month after log1p')
plt.show()
test = pd.read_csv('../input/test.csv')
print(test.head(10))
print(test.isnull().any().describe())
print(test.isnull().all().describe())

test = pd.merge(test,items, how='left', on=['item_id'])
test=test.drop(['item_name'], axis=1)
print(test.head(10))
print(test.isnull().any().describe())
print(test.isnull().all().describe())


print('这是train的数据————————————————')
print('train.shop_id.min: %d'  % train.shop_id.min())
print('train.shop_id.max: %d'  % train.shop_id.max())
print('train.shop_id.mean: %d' % train.shop_id.mean())
print('train.shop_id.median: %d' % train.shop_id.median())
print('这是test的数据—————————————————')
print('test.shop_id.min: %d'  % test.shop_id.min())
print('test.shop_id.max: %d'  % test.shop_id.max())
print('test.shop_id.mean: %d' % test.shop_id.mean())
print('test.shop_id.median: %d' % test.shop_id.median())
print('————————————————————————')
print(train.head(10))
label=pd.get_dummies(train['shop_id'])
print(label.head(5))

for i in range(60):
    label.rename(columns={label.columns[i]: "shop_id_"+str(i)}, inplace=True)

train_shop_id=train.drop(['shop_id'],axis=1)
train_shop_id=pd.concat([train_shop_id, label],axis=1)

print(train_shop_id.head(5))
"""
train_shop_id2  =train_shop_id.drop(['item_category_id'], axis=1)
train_shop_id2  =train_shop_id2.drop(['item_id'], axis=1)

print(train_shop_id2.head(5))

train_shop_id2 = train_shop_id2.groupby([c for c in train_shop_id2.columns if c not in ['item_cnt_month']], as_index=False)[['item_cnt_month']].sum()#分析聚类

print(train_shop_id2.head(5))
train_shop_id2.to_csv('jsh.csv')
"""
print('这是train的数据————————————————')
print('train.item_category_id.min: %d'  % train.item_category_id.min())
print('train.item_category_id.max: %d'  % train.item_category_id.max())
print('train.item_category_id.mean: %d' % train.item_category_id.mean())
print('train.item_category_id.median: %d' % train.item_category_id.median())
print('这是test的数据—————————————————')
print('test.item_category_id.min: %d'  % test.item_category_id.min())
print('test.item_category_id.max: %d'  % test.item_category_id.max())
print('test.item_category_id.mean: %d' % test.item_category_id.mean())
print('test.item_category_id.median: %d' % test.item_category_id.median())
print('————————————————————————')

label2=pd.get_dummies(train_shop_id['item_category_id'])
print(label2.head(5))

for i in range(84):
    label2.rename(columns={label2.columns[i]: "item_category_id_"+str(i)}, inplace=True)

train_item_category_id=train_shop_id.drop(['item_category_id'],axis=1)
train_shop_id_item_category_id=pd.concat([train_item_category_id, label2],axis=1)

print(train_shop_id_item_category_id.head(5))

test=test.drop(['ID'], axis=1)
test['date_block_num']= 34
test['month']= 11
test['year']= 2015

print(test.isnull().any().describe())
print(test.isnull().all().describe())

train_test=pd.concat([train,test],axis=0)

print(train_test.head(5))

label3=pd.get_dummies(train_test['shop_id'])

for i in range(60):
    label3.rename(columns={label3.columns[i]: "shop_id_"+str(i)}, inplace=True)
    
train_test=train_test.drop(['shop_id'],axis=1)
train_test=pd.concat([train_test, label3],axis=1)

label4=pd.get_dummies(train_test['item_category_id'])

for i in range(84):
    label4.rename(columns={label4.columns[i]: "item_category_id_"+str(i)}, inplace=True)
    
train_test=train_test.drop(['item_category_id'],axis=1)
train_test=pd.concat([train_test, label4],axis=1)

test_shop_id_item_category_id=train_test[train_test['date_block_num']==34]

print(test_shop_id_item_category_id.head(5))


#At last，I want check all the numbers.

test_shop_id_item_category_id=test_shop_id_item_category_id.drop(['item_cnt_month'], axis=1)

print(train_shop_id_item_category_id.isnull().any().describe())
#print(train_shop_id_item_category_id.isnull().any())
print(test_shop_id_item_category_id.isnull().any().describe())
#print(test_shop_id_item_category_id.isnull().any())

#test_shop_id_item_category_id.to_csv('the final test.csv',index=False)




import xgboost as xgb
from sklearn.metrics import mean_squared_error#from sklearn.metrics.


#train1=trtrain_month_label=train_shop_id_item_category_id['']ain_shop_id_item_category_id[train_shop_id_item_category_id[date_block_num]<10]  

xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=10, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)
#log1p
"""
x1=train_shop_id_item_category_id[train_shop_id_item_category_id['date_block_num']<10]
y1=np.log1p(x1['item_cnt_month'].clip(0.,20.))
x1=x1.drop(['item_cnt_month'], axis=1)



x1vf=train_shop_id_item_category_id[train_shop_id_item_category_id['date_block_num']==10]
y1vf=np.log1p(x1vf['item_cnt_month'].clip(0.,20.))
x1vf=x1vf.drop(['item_cnt_month'], axis=1)
#print(x1vf.head(5))
#print(y1vf.head(5))

print('the first trainning ......')
xlf.fit(x1,y1)
print('the first predecting ......')
print('FIRST RMSE:', np.sqrt(mean_squared_error(y1vf.clip(0.,20.),xlf.predict(x1vf).clip(0.,20.))))

del x1
del y1
del x1vf
del y1vf

x2=train_shop_id_item_category_id[train_shop_id_item_category_id['date_block_num']<22]
y2=np.log1p((x2['item_cnt_month'].clip(0.,20.)))
x2 = x2.drop(['item_cnt_month'], axis=1)

x2vf=train_shop_id_item_category_id[train_shop_id_item_category_id['date_block_num']==22]
y2vf=np.log1p(x2vf['item_cnt_month'].clip(0.,20.))
x2vf=x2vf.drop(['item_cnt_month'], axis=1)

#print(x2.head(5))
#print(y2.head(5))
#print(x2vf.head(5))
#print(x2vf.head(5))
            
print('the second trainning ......')
xlf.fit(x2,y2)
print('the second predecting ......')
print('FIRST RMSE:', np.sqrt(mean_squared_error(y2vf.clip(0.,20.),xlf.predict(x2vf).clip(0.,20.))))

del x2
del y2
del x2vf
del y2vf
"""
train_shop_id_item_category_id=train_shop_id_item_category_id[train_shop_id_item_category_id['date_block_num']>23]

#train_month_label=train_shop_id_item_category_id['item_cnt_month'].clip(0.,20.)
train_month_label=np.log1p(train_shop_id_item_category_id['item_cnt_month'].clip(0.,20.))
train_shop_id_item_category_id=train_shop_id_item_category_id.drop(['item_cnt_month'], axis=1)

print('trainning ......')
xlf.fit(train_shop_id_item_category_id,train_month_label)

print('predicting ......')
y_predprob = xlf.predict(test_shop_id_item_category_id)

y_predprob=y_predprob.clip(0.,20.)#
#y_predpro=np.expm1(y_predprob)

print('titling.....')
results = pd.Series(y_predprob,name="item_cnt_month")

print('concating.....')
submission = pd.concat([pd.Series(range(0,214200),name = "ID"),results],axis = 1)
print('saving.....')
submission.to_csv("xgr one hot20180424_2 .csv",index=False)



