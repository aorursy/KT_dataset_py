# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#加载数据分析常用库
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# matplotlib inline
import warnings 
warnings.filterwarnings('ignore')
#定义常量
rnn_unit=10       #hidden layer units
input_size=37    
output_size=1
lr=0.0006         #学习率
tf.reset_default_graph()
data = []
# def get_data(batch_size=60,time_step=61,train_begin=0,train_end=139):
#     batch_index=[]
        
#     scaler_for_x=MinMaxScaler(feature_range=(-1,1))  #按列做minmax缩放
#     scaler_for_y=MinMaxScaler(feature_range=(-1,1))
# #     print(data)
#     scaled_x_data=scaler_for_x.fit_transform(data[:,1:])
# #     scaled_y_data=scaler_for_y.fit_transform(data[:,0:1])
#     scaled_y_data = data[:,0]
    
#     label_train = scaled_y_data[train_begin:train_end]
#     label_test = scaled_y_data[train_end:]
#     normalized_train_data = scaled_x_data[train_begin:train_end]
#     normalized_test_data = scaled_x_data[train_end:]
    
#     train_x,train_y=[],[]   #训练集x和y初定义
#     for i in range(len(normalized_train_data)-time_step):
#         if i % batch_size==0:
#             batch_index.append(i)
#         x=normalized_train_data[i:i+time_step,:input_size]
#         y=label_train[i:i+time_step,np.newaxis]
#         train_x.append(x.tolist())
#         train_y.append(y.tolist())
#     batch_index.append((len(normalized_train_data)-time_step))
    
#     size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
#     test_x,test_y=[],[]  
#     for i in range(size-1):
#         x=normalized_test_data[i*time_step:(i+1)*time_step,:input_size]
#         y=label_test[i*time_step:(i+1)*time_step]
#         test_x.append(x.tolist())
#         test_y.extend(y)
# #     test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
# #     test_y.extend((label_test[(i+1)*time_step:]).tolist())    
    
#     return batch_index,train_x,train_y,test_x,test_y,scaler_for_y

# #——————————————————定义神经网络变量——————————————————
# def lstm(X):  
#     batch_size=tf.shape(X)[0]
#     time_step=tf.shape(X)[1]
#     w_in=weights['in']
#     b_in=biases['in']  
#     input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
#     input_rnn=tf.matmul(input,w_in)+b_in
#     input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
#     cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
#     #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)
#     init_state=cell.zero_state(batch_size,dtype=tf.float32)
#     output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
#     output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
#     w_out=weights['out']
#     b_out=biases['out']
#     pred=tf.matmul(output,w_out)+b_out
#     return pred,final_states

# #——————————————————训练模型——————————————————
# def train_lstm(batch_size=80,time_step=61,train_begin=0,train_end=139):
#     X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
#     Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
#     batch_index,train_x,train_y,test_x,test_y,scaler_for_y = get_data(batch_size,time_step,train_begin,train_end)
# #     print(test_x)
#     pred,_=lstm(X)
#     #损失函数
#     loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
#     train_op=tf.train.AdamOptimizer(lr).minimize(loss)  
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         #重复训练5000次
#         iter_time = 1000
#         for i in range(iter_time):
#             for step in range(len(batch_index)-1):
#                 _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
#             if i == 999:    
#                 print('iter:',i,'loss:',loss_)
#         ####predict####
#         test_predict=[]
#         for step in range(len(test_x)):
# #             print(test_x[step])
#             prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
#             predict=prob.reshape((-1))
#             test_predict.extend(predict)
            
# #         test_predict = scaler_for_y.inverse_transform(test_predict)
# #         test_y = scaler_for_y.inverse_transform(test_y)
#         rmse=np.sqrt(mean_squared_error(test_predict,test_y))
#         mae = mean_absolute_error(y_pred=test_predict,y_true=test_y)
#         tmape = np.sum(np.abs((np.array(test_predict)-np.array(test_y))/(1.5-np.array(test_y))))/len(test_y)
#         print ('mae:',mae,'   rmse:',rmse, '         tmape:', tmape, '      score:',(2/(2+mae+tmape))*(2/(2+mae+tmape)))
#     return test_predict
test_correlation = pd.read_csv('../input/train_correlation.csv',encoding='gbk')
test_fund_benchmark_return = pd.read_csv('../input/train_fund_benchmark_return.csv',encoding='gbk')
index_return = pd.read_csv('../input/train_index_return.csv',encoding='gbk')
test_correlation.index = test_correlation['Unnamed: 0']
test_fund_benchmark_return.index = test_fund_benchmark_return['Unnamed: 0']
index_return.index = index_return['Unnamed: 0']
cor_data = test_correlation.T
fund_ben_data = test_fund_benchmark_return.T
index_data = index_return.T
cor_data = cor_data.drop(['Unnamed: 0'])
fund_ben_data = fund_ben_data.drop(['Unnamed: 0'])
index_data = index_data.drop(['Unnamed: 0'])

del test_fund_benchmark_return
del test_correlation
col_list = cor_data.columns
fund_list = list(fund_ben_data.columns)
fund_list[0] = 'fund_ID'
X = []
y = []
for i in col_list[:200]:
    data = cor_data[[i]]
    funds = i.split('-')
    fund1 = funds[0].strip()
    fund2 = funds[1].strip()
    fund_data = fund_ben_data[[fund1]]
    data = data.merge(fund_data, left_index=True, right_index=True, how='left')
    fund_data = fund_ben_data[[fund2]]
    data = data.merge(fund_data, left_index=True, right_index=True, how='left')
    data = data.merge(index_data, left_index=True, right_index=True, how='left')
    data = data.iloc[:,:].values
    data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
#     print(data.shape)
    temp_X = []
    for k in range(0, data.shape[0]):
        if k%5!=0: 
            continue
        temp_X=temp_X+list(data[k,1:])
        if(k > 61):
            for j in range(1, data.shape[1]):
                temp_X.pop(0)
#             print(len(temp_X))
            X.append(temp_X)
            y.append(data[k,0])
    if(len(X)%1000 == 0):
        print(len(X))
        print(len(y))
print(len(X))
from sklearn.ensemble import AdaBoostRegressor
clf = AdaBoostRegressor(learning_rate=0.05,n_estimators=200)
clf.fit(X[0:1000],y[0:1000])
pre = clf.predict(X[11000:12500])
mae = np.sum(np.abs(np.array(pre) - np.array(y[11000:12500])))/len(pre)
tmape = np.sum(np.abs((np.array(pre) - np.array(y[11000:12500]))/(1.5-np.array(y[11000:12500]))))/len(pre)
score = (2/(2+mae+tmape))*(2/(2+mae+tmape))
print('mae:', mae, '     tmape:', tmape, '      score:', score)
test_correlation_1 = pd.read_csv('../input/test_correlation.csv',encoding='gbk')
test_fund_benchmark_return_1 = pd.read_csv('../input/test_fund_benchmark_return.csv',encoding='gbk')
index_return_1 = pd.read_csv('../input/test_index_return.csv',encoding='gbk')
test_correlation_1.index = test_correlation_1['Unnamed: 0']
test_fund_benchmark_return_1.index = test_fund_benchmark_return_1['Unnamed: 0']
index_return_1.index = index_return_1['Unnamed: 0']
cor_data_1 = test_correlation_1.T
fund_ben_data_1 = test_fund_benchmark_return_1.T
index_data_1 = index_return_1.T
cor_data_1 = cor_data_1.drop(['Unnamed: 0'])
fund_ben_data_1 = fund_ben_data_1.drop(['Unnamed: 0'])
index_data_1 = index_data_1.drop(['Unnamed: 0'])

del test_fund_benchmark_return_1
del test_correlation_1
col_list_1 = cor_data_1.columns
fund_list_1 = list(fund_ben_data_1.columns)
fund_list_1[0] = 'fund_ID'
cor_data_2 = pd.concat([cor_data, cor_data_1])
fund_ben_data_2 = pd.concat([fund_ben_data, fund_ben_data_1])
col_list = cor_data_2.columns
cor_data_2.head()
# X = []
# y = []
# for i in col_list[:200]:
#     data = cor_data_2[[i]]
#     funds = i.split('-')
#     fund1 = funds[0].strip()
#     fund2 = funds[1].strip()
#     fund_data = fund_ben_data_2[[fund1]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     fund_data = fund_ben_data[[fund2]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     data = data.merge(index_data, left_index=True, right_index=True, how='left')
#     data = data.iloc[:,:].values
#     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
# #     print(data.shape)
#     temp_X = []
#     for k in range(0, data.shape[0]):
#         if k%5!=0: 
#             continue
#         temp_X=temp_X+list(data[k,1:])
#         if(k > 61):
#             for j in range(1, data.shape[1]):
#                 temp_X.pop(0)
# #             print(len(temp_X))
#             X.append(temp_X)
#             y.append(data[k,0])
#     if(len(X)%1000 == 0):
#         print(len(X))
#         print(len(y))
# print(len(X))
test_X = []
for i in col_list_1[:]:
    data = cor_data_1[[i]]
    funds = i.split('-')
    fund1 = funds[0].strip()
    fund2 = funds[1].strip()
    fund_data = fund_ben_data_1[[fund1]]
    data = data.merge(fund_data, left_index=True, right_index=True, how='right')
    fund_data = fund_ben_data_1[[fund2]]
    data = data.merge(fund_data, left_index=True, right_index=True, how='left')
    data = data.merge(index_data_1, left_index=True, right_index=True, how='left')
    data = data.iloc[:,:].values
    data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
#     print(data.shape)
    temp_X = []
    for k in range(0, data.shape[0]):
        if k%5!=0: 
            continue
        temp_X=temp_X+list(data[k,1:])
        if(k > 61):
            for j in range(1, data.shape[1]):
                temp_X.pop(0)
#             print(len(temp_X))
    test_X.append(temp_X)
    if(len(test_X)%1000 == 0):
        print(len(test_X))
pre = clf.predict(test_X)
# LSTM 不可行
# pre = []
# for i in col_list[:100]:
#     tf.reset_default_graph()
#     #输入层、输出层权重、偏置
#     weights={
#              'in':tf.Variable(tf.random_normal([input_size,rnn_unit],dtype=tf.float32)),
#              'out':tf.Variable(tf.random_normal([rnn_unit,1]),dtype=tf.float32)
#              }
#     biases={
#             'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,],dtype=tf.float32)),
#             'out':tf.Variable(tf.constant(0.1,shape=[1,],dtype=tf.float32))
#             }
# #     print(cor_data.head())
#     data = cor_data[[i]]
# #     print(data.head())
#     funds = i.split('-')
#     fund1 = funds[0].strip()
#     fund2 = funds[1].strip()
#     fund_data = fund_ben_data[[fund1]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     fund_data = fund_ben_data[[fund2]]
#     data = data.merge(fund_data, left_index=True, right_index=True, how='left')
#     data = data.merge(index_data, left_index=True, right_index=True, how='left')
#     data = data.iloc[:,:].values
# #     print(data[:,0])
# #     where_are_nan = np.isnan(data[:,0])
#     data[pd.isnull(np.array(data[:,0], dtype=float)),0] = 0
#     print(data.shape)
# #     print(data)
#     test_predict = train_lstm(batch_size=80,time_step=61,train_begin=1,train_end=139)
#     pre.append(test_predict[-1])
#     if len(pre) % 100 == 0:
#         print(len(pre))
pre
result = pd.DataFrame(columns=['ID', '2018-03-19'])
result['ID'] = col_list[:]
result['2018-03-19'] = pre
result.to_csv('result2.csv', header=True,index=False)
# plt.figure(figsize=(24,8))
# plt.plot(data[:, -1])
# plt.plot([None for _ in range(487)] + [x for x in test_predict])
# plt.show()

