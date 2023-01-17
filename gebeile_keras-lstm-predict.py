# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -*- coding: utf-8 -*-



import numpy as np

import pandas as pd

# from  pyecharts import *

import os

from keras.layers import Dense

from keras.layers import LSTM

from keras.models import Sequential, load_model

from sklearn.model_selection import  train_test_split

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import collections

from math import sqrt

from sklearn.metrics import mean_squared_error





# 数据探查

train_data=pd.read_csv(r'/kaggle/input/demand-forecasting-kernels-only/train.csv',engine='python')

train_columns=train_data.columns #[date,store,item,sales],data:1826,2013-01-01,2017-12-31;store:1-10;item:1-50;



#判断数据中是否含有空置

isnull_data=train_data[train_data.isnull().T.any()]



#判断每个店铺不同商品的异常值

for i,item in enumerate(list(set(train_data['store']))):

    for j,key in enumerate(list(set(train_data['item']))):

        percentile=np.percentile(train_data[(train_data['store']==item)&(train_data['item']==key)]['sales'],[0,25,50,75,100])

        #q1-1.5*iqr,q3+1.5*iqr 异常点

        low=percentile[1]-1.5*(percentile[3]-percentile[1])

        up=percentile[3]+1.5*(percentile[3]-percentile[1])

        #num=train_data[(train_data['store']==item)&(train_data['item']==key)&((train_data['sales']>up)|(train_data['sales']<low))]['sales'].count()

        train_data.loc[(train_data['store']==item)&(train_data['item']==key)&(train_data['sales']>up),'sales']=up

        train_data.loc[(train_data['store']==item)&(train_data['item']==key)&(train_data['sales']<low),'sales']=low



# train_data=train_data[~train_data.index.isin(diff_total_percentile_index)]

# train_data=train_data.reset_index(drop=True)

# train_data=pd.DataFrame(train_data,columns=['date', 'store', 'item', 'sales'])

print(train_data.dtypes)





#生成训练数据集，data:DataFrame,timestep：记忆时间跨度

def creata_dataset(data,timestep):

    x_data,y_data=[],[]

    for index in range(len(data)-timestep-1):

        x_data.append(data[index:index + timestep])

        y_data.append(data[index + timestep])



    return np.array(x_data),np.array(y_data)

##########以总的门店为例,单变量测试分析################

train_data_1=pd.DataFrame(train_data['sales'],columns=['sales'])



trainsize=int(len(train_data_1)*0.7)

train_data_1=train_data_1.values

train_data_1 = train_data_1.astype('float32')



#归一化 在下一步会讲解

scaler = MinMaxScaler(feature_range=(0, 1))

train_data_1 = scaler.fit_transform(train_data_1)



x_train,y_train=creata_dataset(train_data_1[:trainsize],5)

x_test,y_test=creata_dataset(train_data_1[trainsize:],5)



print(x_train,y_train)



x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))



if os.path.exists('/kaggle/input/kernel3f8d042026/Test_md1.h5'):

    model = load_model("/kaggle/input/kernel3f8d042026/Test_md1.h5")

else:

    model = Sequential()

    model.add(LSTM(50, input_shape=(None,1)))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=100,batch_size=100, verbose=2)

    model.save("./Test_md1.h5")   

print('***********训练数据训练已完成*******************')

testPredict = model.predict(x_test)

trainPredict = model.predict(x_train)



#反归一化

trainPredict = scaler.inverse_transform(trainPredict)

y_train = scaler.inverse_transform(y_train)

testPredict = scaler.inverse_transform(testPredict)

y_test = scaler.inverse_transform(y_test)





#以门店1为例测试值与预测值

# line2=Line('门店1的预测值与真实值对比情况')

# line2.add('真实值',[i for i in range(len(y_test))],[item[0] for item in y_test.tolist()],is_datazoom_show=True,tooltip_tragger='axis')

# line2.add('测试值',[i for i in range(len(testPredict))],[item[0] for item in testPredict.tolist()],is_datazoom_show=True,tooltip_tragger='axis')

# line2.render('门店1的预测值与真实值对比情况.html')



plt.plot(y_train)

plt.plot(trainPredict[1:])

plt.show()

plt.plot(y_test)

plt.plot(testPredict[1:])

plt.show()



#测试数据预测

test_data=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv',engine='python')

test_data=pd.DataFrame(test_data['sales'],columns=['sales'])

test_data=test_data.values

test_data=test_data.astype('float32')



scaler=MinMaxScaler(feature_range=(0,1))

test_data=scaler.fit_transform(test_data)

test_data=test_data.reshape((test_data.shape[0],test_data.shape[1],1))

test_predict=model.predict(test_data)



#test_predict=test_data.reshape((test_predict.shape[0],test_predict[2]))

test_predict=scaler.inverse_transform(test_predict)



y_test=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv',engine='python')

y_test=pd.DataFrame(y_test['sales'],columns=['sales'])

y_submission=y_test.values

val_loss=sqrt(mean_squared_error(y_submission,test_predict))

pd.DataFrame(test_predict,columns=['sales']).to_csv('test_predict.csv',encoding='utf-8')

print('***********测试数据预测已完成,均方误差：%f*******************'%val_loss)
##########以总的门店为例,多变量测试分析################



#查看总的时间所有门店销售情况

# Line=Line('散点图','所有门店总体销售数')

# date=train_data[(train_data['store']==1) & (train_data['item']==1)]['date']

# sales=train_data[(train_data['store']==1) & (train_data['item']==1)]['sales']

# Line.add('2013-2017年总体销售数量',date,sales,xaxis_name='日期',yaxis_name='销售量',is_datazoom_show=True,tooltip_tragger='axis',legend_text_size=10)

# Line.render('所有门店总体销售数走势图.html')



#查看每个门店的销售情况

# shop_sales=train_data.groupby(['date','store'],as_index=False)['sales'].sum() #生成data_groupl类型数据,需循环计算

shops=set(train_data['store'])

# line2=Line('不同门店的销售情况')

# print(shop_sales[shop_sales['store']==1])

# for shop in shops:

#     line2.add('商家%s'%shop,shop_sales[shop_sales['store']==shop]['date'],shop_sales[shop_sales['store']==shop]['sales'],is_datazoom_show=True,tooltip_tragger='axis')

# line2.render('不同门店的销售情况.html')



#查看不同门店不同商品销售情况

# items=set(train_data['item'])



# for shop in shops:

#     line3=Line('门店%s不同商品的销售情况'%shop)

#     for item in items:

#         line3.add('商品%s'%item,train_data[(train_data['item']==item)&(train_data['store']==shop)]['date'],train_data[(train_data['item']==item)&(train_data['store']==shop)]['sales'],is_datazoom_show=True,tooltip_tragger='axis')

#     line3.render('门店%s不同商品的销售情况.html'%shop)



#计算每个商品的每月平均销售量

train_data['mouth']=train_data['date'].str.extract('\d+-(\d+)-\d+')

diff_item_mouth_avg_sales=train_data.groupby(['mouth','item'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_item_mouth_avg_sales'})

train_data=pd.merge(train_data,diff_item_mouth_avg_sales,how='left',on=['mouth','item'])

print('*****训练数据：每个商品的每月平均销售量*****,计算已完成')



#计算不同商店不同商品每月平均销售量

diff_shop_item_mouth_avg_sales=train_data.groupby(['mouth','item','store'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_shop_item_mouth_avg_sales'})

train_data=pd.merge(train_data,diff_shop_item_mouth_avg_sales,how='left',on=['mouth','item','store'])

print('****训练数据：不同商店不同商品每月平均销售量*****,计算已完成')



#计算不同商店每月平均销售量

diff_shop_mouth_avg_sales=train_data.groupby(['mouth','store'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_shop_mouth_avg_sales'})

train_data=pd.merge(train_data,diff_shop_mouth_avg_sales,how='left',on=['mouth','store'])

print('****训练数据：不同商店每月平均销售量*****,计算已完成')



#计算不同商店每天的销售量

train_data['day']=train_data['date'].str.extract('\d+-(\d+-\d+)')

diff_shop_day_avg_sales=train_data.groupby(['day','store'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_shop_day_avg_sales'})

train_data=pd.merge(train_data,diff_shop_day_avg_sales,how='left',on=['day','store'])

print('****训练数据：不同商店每天的销售量*****,计算已完成')



#计算不同商店不同商品每天平均销售量

diff_shop_item_day_avg_sales=train_data.groupby(['day','item','store'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_shop_item_day_avg_sales'})

train_data=pd.merge(train_data,diff_shop_item_day_avg_sales,how='left',on=['day','item','store'])

print('****训练数据：不同商店不同商品每天平均销售量*****,计算已完成')



#计算每个商品的每天平均销售量

diff_item_day_avg_sales=train_data.groupby(['day','item'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_item_day_avg_sales'})

train_data=pd.merge(train_data,diff_item_day_avg_sales,how='left',on=['day','item'])

print('****训练数据：每个商品的每天平均销售量*****,计算已完成')



#计算每个商店的每个日期平均销量

diff_store_date_avg_sales=train_data.groupby(['date','store'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_store_date_avg_sales'})

train_data=pd.merge(train_data,diff_store_date_avg_sales,how='left',on=['date','store'])

print('****训练数据：每个商店的每个日期平均销量*****,计算已完成')



#计算每个商店每个商品不同季度的销售量

def reg_quarter(mouth):

    if int(mouth)<=3:

        return 1

    elif int(mouth)<=6:

        return 2

    elif int(mouth)<=9:

        return 3

    else:

        return 4



train_data['quarter']=train_data['mouth'].apply(reg_quarter)

diff_store_item_quarter_avg_sales=train_data.groupby(['quarter','item','store'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_store_item_quarter_avg_sales'})

train_data=pd.merge(train_data,diff_store_item_quarter_avg_sales,how='left',on=['quarter','item','store'])

print('****训练数据：每个商店每个商品不同季度的销售量*****,计算已完成')



#计算每个商店不同季度的销售量

diff_store_quarter_avg_sales=train_data.groupby(['quarter','store'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_store_quarter_avg_sales'})

train_data=pd.merge(train_data,diff_store_quarter_avg_sales,how='left',on=['quarter','store'])

print('****训练数据：每个商店不同季度的销售量*****,计算已完成')



#计算每个商品不同季度的销售量

diff_item_quarter_avg_sales=train_data.groupby(['quarter','item'],as_index=False)['sales'].mean().rename(columns={'sales':'diff_item_quarter_avg_sales'})

train_data=pd.merge(train_data,diff_item_quarter_avg_sales,how='left',on=['quarter','item'])

print('****训练数据：每个商品不同季度的销售量*****,计算已完成')



#将商店以及商品one-hot稀疏化



from sklearn.preprocessing import OneHotEncoder

#将store稀疏化(one-hot)

def scartter_type_jhcz(name:str,num:int):

    store_sales=[]

    for i in range(len(set(train_data[name]))):

        sales=train_data[train_data[name]==(i+1)]['sales'].T

        store_sales.append(list(sales))



    scartter=MinMaxScaler(feature_range=(0,1))

    store_sales=scartter.fit_transform(store_sales)

    model=KMeans(n_clusters=num)

    model.fit(store_sales)

    y=model.predict(store_sales)

    store_sales=scartter.inverse_transform(store_sales)

    store_class=collections.defaultdict(lambda:[])

    store_data=collections.defaultdict(lambda:[])

    for i,item in enumerate(y):

        store_class[item].append(i+1)

        store_data[item].append(store_sales[i])



    return store_class,store_data,y

store_class,store_data,y=scartter_type_jhcz('store',2)

item_class,item_data,y=scartter_type_jhcz('item',3)

#item_class:{1: [3, 4, 5, 16, 17, 23, 27, 34, 37, 40, 41, 42, 44, 47, 49], 2: [6, 7, 9, 14, 19, 20, 21, 26, 30, 31, 32, 39, 43, 46, 48], 0: [10, 11, 12, 13, 15, 18, 22, 24, 25, 28, 29, 33, 35, 36, 38, 45, 50]}) 

plt.scatter(np.array(item_data[0])[:,0],np.array(item_data[0])[:,1])

plt.scatter(np.array(item_data[1])[:,0],np.array(item_data[1])[:,1])

plt.scatter(np.array(item_data[2])[:,0],np.array(item_data[2])[:,1])

plt.show()

train_data['type']=0

for i,item in enumerate(item_class.keys()):

    train_data.loc[train_data['item'].isin(item_class[item]),'type']=item #这样赋值怎末不成功？



train_data['class']=0

for i,item in enumerate(store_class.keys()):

    train_data.loc[train_data['store'].isin(store_class[item]),'class']=item

    

df_item_dummy=pd.get_dummies(train_data['type'],prefix='item')

train_data=pd.concat([train_data,df_item_dummy],axis=1)



# df_store_dummy=pd.get_dummies(train_data['class'],prefix='store')

# train_data=pd.concat([train_data,df_store_dummy],axis=1)



print(set(df_item_dummy),set(train_data['type']),train_data.dtypes,train_data[train_data['item'].isin(item_class[1])]['type'])

dataset=pd.DataFrame(train_data[['class','item_0','item_1','item_2','diff_item_mouth_avg_sales','diff_shop_item_mouth_avg_sales'

,'diff_shop_mouth_avg_sales','diff_shop_day_avg_sales','diff_shop_item_day_avg_sales'

,'diff_item_day_avg_sales','diff_store_item_quarter_avg_sales','diff_store_quarter_avg_sales','diff_item_quarter_avg_sales','sales']],columns=['class','item_0','item_1','item_2','diff_item_mouth_avg_sales','diff_shop_item_mouth_avg_sales'

,'diff_shop_mouth_avg_sales','diff_shop_day_avg_sales','diff_shop_item_day_avg_sales'

,'diff_item_day_avg_sales','diff_store_item_quarter_avg_sales','diff_store_quarter_avg_sales','diff_item_quarter_avg_sales','sales'])



# dataset=pd.DataFrame(train_data[(train_data['store']==1)&(train_data['item']==1)][['store_1','store_2','store_3','store_4','store_5','store_6','store_7','store_8','store_9','store_10'

# ,'item_0','item_1','item_2','diff_item_day_avg_sales','diff_store_date_avg_sales','sales']])

# dataset.to_csv('dataset.csv')

# print(dataset.to_csv('dataset.csv'))

dataset=dataset.values

dataset=dataset.astype('float32')



#特征选择

# from sklearn.feature_selection import SelectKBest

# from sklearn.feature_selection import chi2

# test = SelectKBest(score_func=chi2, k=8)

# X=dataset[:,:-1]

# Y=dataset[:,-1]

# fit = test.fit(X, Y)

# print(fit.scores_)



#所有变量归一化:仅对特征变量进行归一化，减少特征之间的差异

scaler=MinMaxScaler(feature_range=(0,1))

dataset_feature=scaler.fit_transform(dataset[:,:-1])



#切分训练集与测试集

trainsize=int(len(dataset_feature)*0.65)

train_list=dataset_feature[:trainsize,:]

x_train,y_train=train_list[:,:],dataset[:trainsize,-1].reshape(-1, 1)



test_list=dataset_feature[trainsize:,:]

x_test,y_test=test_list[:,:],dataset[trainsize:,-1].reshape(-1, 1)



#reshape==>[samples,timestep,features]

x_train=x_train.reshape((x_train.shape[0],1,x_train.shape[1]))

x_test=x_test.reshape((x_test.shape[0],1,x_test.shape[1]))



print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)



if os.path.exists('./all_store.h5'):

    model=load_model('./all_store.h5')

else:

    #运行报错：numpy.linalg.LinAlgError: SVD did not converge;1.存在异常数据：NAN INF; 2.数据量太大导致内存不足报错

    #design network

    model=Sequential()

    model.add(LSTM(100,input_shape=(x_train.shape[1],x_train.shape[2])))

    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')

    history =model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_test, y_test), verbose=2, shuffle=False)

    model.save('all_store.h5')



    # plot history

    plt.plot(history.history['loss'], label='train')

    plt.plot(history.history['val_loss'], label='test')

    plt.legend()

    plt.show()



# make a prediction

test_predict = model.predict(x_test)

# x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))

# # invert scaling for forecast

# # inv_yhat = np.concatenate((test_predict, x_test[:, 1:]), axis=1)

# inv_yhat = np.concatenate((test_predict, x_test), axis=1)

# inv_yhat = scaler.inverse_transform(inv_yhat)

# inv_yhat = inv_yhat[:,0]

# # invert scaling for actual

# inv_y=np.concatenate((y_test,x_test),axis=1)

# inv_y = scaler.inverse_transform(inv_y)

# inv_y = inv_y[:,0]

rmse = sqrt(mean_squared_error(test_predict, y_test))

print('Test RMSE: %.3f' % rmse)



##########预测分析################

predict_data=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv',engine='python')

#计算预测数据的衍生变量



#计算每个商品的每月平均销售量

predict_data['mouth']=predict_data['date'].str.extract('\d+-(\d+)-\d+')

predict_data=pd.merge(predict_data,diff_item_mouth_avg_sales,how='left',on=['mouth','item'])

print('*****预测数据：每个商品的每月平均销售量*****,计算已完成')



#计算不同商店不同商品每月平均销售量

predict_data=pd.merge(predict_data,diff_shop_item_mouth_avg_sales,how='left',on=['mouth','item','store'])



print('****预测数据：不同商店不同商品每月平均销售量*****,计算已完成')



#计算不同商店每月平均销售量

predict_data=pd.merge(predict_data,diff_shop_mouth_avg_sales,how='left',on=['mouth','store'])

print('****预测数据：不同商店每月平均销售量*****,计算已完成')



#计算不同商店每天的销售量

predict_data['day']=predict_data['date'].str.extract('\d+-(\d+-\d+)')

predict_data=pd.merge(predict_data,diff_shop_day_avg_sales,how='left',on=['day','store'])

print('****预测数据：不同商店每天的销售量*****,计算已完成')



#计算不同商店不同商品每天平均销售量

predict_data=pd.merge(predict_data,diff_shop_item_day_avg_sales,how='left',on=['day','item','store'])

print('****预测数据：不同商店不同商品每天平均销售量*****,计算已完成')



#计算每个商品的每天平均销售量

predict_data=pd.merge(predict_data,diff_item_day_avg_sales,how='left',on=['day','item'])

print('****预测数据：每个商品的每天平均销售量*****,计算已完成')



#计算每个商店每个商品不同季度的销售量

predict_data['quarter']=predict_data['mouth'].apply(reg_quarter)

predict_data=pd.merge(predict_data,diff_store_item_quarter_avg_sales,how='left',on=['quarter','item','store'])

print('****训练数据：每个商店每个商品不同季度的销售量*****,计算已完成')



#计算每个商店不同季度的销售量

predict_data=pd.merge(predict_data,diff_store_quarter_avg_sales,how='left',on=['quarter','store'])

print('****训练数据：每个商店不同季度的销售量*****,计算已完成')



#计算每个商品不同季度的销售量

predict_data=pd.merge(predict_data,diff_item_quarter_avg_sales,how='left',on=['quarter','item'])

print('****训练数据：每个商品不同季度的销售量*****,计算已完成')



#计算每个商店的每个日期平均销量

# predict_data=pd.merge(predict_data,diff_store_date_avg_sales,how='left',on=['date','store'])

# print('****预测数据：每个商店的每个日期平均销量*****,计算已完成')



#商店one-hot

# predict_store_onehot=pd.get_dummies(predict_data['store'],prefix='store')

# predict_data=pd.concat([predict_store_onehot,predict_data],axis=1)

predict_data['class']=0

for i,item in enumerate(store_class.keys()):

    predict_data.loc[predict_data['store'].isin(store_class[item]),'class']=item



#商品one-hot

predict_data['type']=0

for i,item in enumerate(item_class.keys()):

    predict_data.loc[predict_data['item'].isin(item_class[item]),'type']=item





predict_item_onehot=pd.get_dummies(predict_data['type'],prefix='item')

predict_data=pd.concat([predict_item_onehot,predict_data],axis=1)



predict_data=pd.DataFrame(predict_data[['class','item_0','item_1','item_2','diff_item_mouth_avg_sales','diff_shop_item_mouth_avg_sales'

,'diff_shop_mouth_avg_sales','diff_shop_day_avg_sales','diff_shop_item_day_avg_sales'

,'diff_item_day_avg_sales','diff_store_item_quarter_avg_sales','diff_store_quarter_avg_sales','diff_item_quarter_avg_sales']],columns=['class','item_0','item_1','item_2','diff_item_mouth_avg_sales','diff_shop_item_mouth_avg_sales'

,'diff_shop_mouth_avg_sales','diff_shop_day_avg_sales','diff_shop_item_day_avg_sales'

,'diff_item_day_avg_sales','diff_store_item_quarter_avg_sales','diff_store_quarter_avg_sales','diff_item_quarter_avg_sales'])



predict_data=predict_data.values

scaler=MinMaxScaler(feature_range=(0,1))

predict_data=scaler.fit_transform(predict_data)



predict_data=predict_data.reshape((predict_data.shape[0],1,predict_data.shape[1]))

print(predict_data.shape)

predict_y=model.predict(predict_data)

# predict_data=predict_data.reshape((predict_data.shape[0],predict_data.shape[2]))

# predict_y=np.concatenate((predict_y,predict_data[:,1:]),axis=1)

# predict_y=scaler.inverse_transform(predict_y)

# print(predict_y[:5,:])

# predict_y=predict_y[:,0]

val_loss=sqrt(mean_squared_error(y_submission,predict_y))

print('**************多变量损失：%f*******************'%val_loss)

predict_y=pd.DataFrame(predict_y,columns=['sales'])

predict_y['id']=predict_y.index

pd.DataFrame(predict_y[['id','sales']],columns=['id','sales']).to_csv('submission.csv',index=False,encoding='utf-8')