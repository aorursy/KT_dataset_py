#-*- coding:utf-8 -*-

import numpy as np 

import pandas as pd

import warnings  

warnings.filterwarnings('ignore')



import os

#输出KDD99中所有数据文件

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



seed = 51
#读入10%的训练子集

data = pd.read_csv('/kaggle/input/kdd-cup-1999-data/kddcup.data_10_percent_corrected', header=None)
#每条连接记录的42项，最后一项为标签

data.columns = [

    'duration',#持续时间，范围是 [0, 58329] 

    'protocol_type',#协议类型，三种：TCP, UDP, ICMP

    'service',#目标主机的网络服务类型，共有70种，如‘http_443′,‘http_8001′,‘imap4′等

    'flag',#连接正常或错误的状态，离散类型，共11种，如‘S0′,‘S1′,‘S2′等

    'src_bytes',#从源主机到目标主机的数据的字节数，范围是 [0,1379963888]

    'dst_bytes',#从目标主机到源主机的数据的字节数，范围是 [0.1309937401]

    'land',#若连接来自/送达同一个主机/端口则为1，否则为0

    'wrong_fragment',#错误分段的数量，连续类型，范围是[0,3]

    'urgent',#加急包的个数，连续类型，范围是[0,14]

    'hot',#访问系统敏感文件和目录的次数，范围是[0,101]

    'num_failed_logins',#登录尝试失败的次数，范围是[0,5]

    'logged_in',#成功登录则为1，否则为0

    'num_compromised',#compromised条件出现的次数，范围是[0,7479]

    'root_shell',#若获得root shell则为1，否则为0

    'su_attempted',#若出现”su root”命令则为1，否则为0

    'num_root',#root用户访问次数，范围是[0,7468]

    'num_file_creations',#文件创建操作的次数，范围是[0,100]

    'num_shells',#使用shell命令的次数，范围是[0,5]

    'num_access_files',#访问控制文件的次数，范围是[0,9]

    'num_outbound_cmds',# 一个FTP会话中出站连接的次数，数据集中这一特征出现次数为0。

    'is_host_login',#登录是否属于“hot”列表，是为1，否则为0

    'is_guest_login',#若是guest 登录则为1，否则为0

    'count',#过去两秒内，与当前连接具有相同的目标主机的连接数，范围是[0,511]

    'srv_count',#过去两秒内，与当前连接具有相同服务的连接数，范围是[0,511]

    'serror_rate',#过去两秒内，在与当前连接具有相同目标主机的连接中，出现“SYN” 错误的连接的百分比，范围是[0.00,1.00]

    'srv_serror_rate',#过去两秒内，在与当前连接具有相同服务的连接中，出现“SYN” 错误的连接的百分比，范围是[0.00,1.00]

    'rerror_rate',#过去两秒内，在与当前连接具有相同目标主机的连接中，出现“REJ” 错误的连接的百分比，范围是[0.00,1.00]

    'srv_rerror_rate',#过去两秒内，在与当前连接具有相同服务的连接中，出现“REJ” 错误的连接的百分比，范围是[0.00,1.00]

    'same_srv_rate',#过去两秒内，在与当前连接具有相同目标主机的连接中，与当前连接具有相同服务的连接的百分比，范围是[0.00,1.00]

    'diff_srv_rate',#过去两秒内，在与当前连接具有相同目标主机的连接中，与当前连接具有不同服务的连接的百分比，范围是[0.00,1.00]

    'srv_diff_host_rate',#过去两秒内，在与当前连接具有相同服务的连接中，与当前连接具有不同目标主机的连接的百分比，范围是[0.00,1.00]

    'dst_host_count',#前100个连接中，与当前连接具有相同目标主机的连接数，范围是[0,255]

    'dst_host_srv_count',#前100个连接中，与当前连接具有相同目标主机相同服务的连接数，范围是[0,255]

    'dst_host_same_srv_rate',#前100个连接中，与当前连接具有相同目标主机相同服务的连接所占的百分比，范围是[0.00,1.00]

    'dst_host_diff_srv_rate',#前100个连接中，与当前连接具有相同目标主机不同服务的连接所占的百分比，范围是[0.00,1.00]

    'dst_host_same_src_port_rate',#前100个连接中，与当前连接具有相同目标主机相同源端口的连接所占的百分比，范围是[0.00,1.00]

    'dst_host_srv_diff_host_rate',#前100个连接中，与当前连接具有相同目标主机相同服务的连接中，与当前连接具有不同源主机的连接所占的百分比，范围是[0.00,1.00]

    'dst_host_serror_rate',#前100个连接中，与当前连接具有相同目标主机的连接中，出现SYN错误的连接所占的百分比，范围是[0.00,1.00]

    'dst_host_srv_serror_rate',#前100个连接中，与当前连接具有相同目标主机相同服务的连接中，出现SYN错误的连接所占的百分比，范围是[0.00,1.00]

    'dst_host_rerror_rate',#dst_host_rerror_rate. 前100个连接中，与当前连接具有相同目标主机的连接中，出现REJ错误的连接所占的百分比，范围是[0.00,1.00]

    'dst_host_srv_rerror_rate',#前100个连接中，与当前连接具有相同目标主机相同服务的连接中，出现REJ错误的连接所占的百分比，范围是[0.00,1.00] 

    'outcome'#标签

]
#展示数据

data.sample(10)
print(data['num_outbound_cmds'].unique())
#将数字值编码为z分数

def encode_numeric_zscore(df, name, mean=None, sd=None):

    if mean is None:

        mean = df[name].mean()



    if sd is None:

        sd = df[name].std()



    df[name] = (df[name] - mean) / sd

    

#将文本值编码为虚拟变量(即[1,0,0],[0,1,0],[0,0,1]代表了红色、绿色、蓝色)

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)
#对每一项数据进行相应处理

encode_numeric_zscore(data, 'duration')

encode_text_dummy(data, 'protocol_type')

encode_text_dummy(data, 'service')

encode_text_dummy(data, 'flag')

encode_numeric_zscore(data, 'src_bytes')

encode_numeric_zscore(data, 'dst_bytes')

encode_text_dummy(data, 'land')

encode_numeric_zscore(data, 'wrong_fragment')

encode_numeric_zscore(data, 'urgent')

encode_numeric_zscore(data, 'hot')

encode_numeric_zscore(data, 'num_failed_logins')

encode_text_dummy(data, 'logged_in')

encode_numeric_zscore(data, 'num_compromised')

encode_numeric_zscore(data, 'root_shell')

encode_numeric_zscore(data, 'su_attempted')

encode_numeric_zscore(data, 'num_root')

encode_numeric_zscore(data, 'num_file_creations')

encode_numeric_zscore(data, 'num_shells')

encode_numeric_zscore(data, 'num_access_files')

encode_numeric_zscore(data, 'num_outbound_cmds')

encode_text_dummy(data, 'is_host_login')

encode_text_dummy(data, 'is_guest_login')

encode_numeric_zscore(data, 'count')

encode_numeric_zscore(data, 'srv_count')

encode_numeric_zscore(data, 'serror_rate')

encode_numeric_zscore(data, 'srv_serror_rate')

encode_numeric_zscore(data, 'rerror_rate')

encode_numeric_zscore(data, 'srv_rerror_rate')

encode_numeric_zscore(data, 'same_srv_rate')

encode_numeric_zscore(data, 'diff_srv_rate')

encode_numeric_zscore(data, 'srv_diff_host_rate')

encode_numeric_zscore(data, 'dst_host_count')

encode_numeric_zscore(data, 'dst_host_srv_count')

encode_numeric_zscore(data, 'dst_host_same_srv_rate')

encode_numeric_zscore(data, 'dst_host_diff_srv_rate')

encode_numeric_zscore(data, 'dst_host_same_src_port_rate')

encode_numeric_zscore(data, 'dst_host_srv_diff_host_rate')

encode_numeric_zscore(data, 'dst_host_serror_rate')

encode_numeric_zscore(data, 'dst_host_srv_serror_rate')

encode_numeric_zscore(data, 'dst_host_rerror_rate')

encode_numeric_zscore(data, 'dst_host_srv_rerror_rate')



#因为每行都存在num_outbound_cmds=0，因此需要滤除缺失数据

data.dropna(inplace=True, axis=1)

#各类别统计展示

data.outcome.value_counts()
#将前41项作为输入数据x，最后一列标签列作为y

x_columns = data.columns.drop('outcome')

x = data[x_columns].values

dummies = pd.get_dummies(data['outcome']) #分类

outcomes = dummies.columns

num_classes = len(outcomes)

y = dummies.values
#随机选取训练集和测试集

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)
import tensorflow

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, ELU, Input, Dropout



input = Input(shape=x.shape[1])

#构造神经网络

m = Dense(64)(input) #全连接层

m = ELU()(m) #激活函数

m = Dropout(0.33)(m) #减轻过拟合



m = Dense(64)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(32)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



m = Dense(16)(m)

m = ELU()(m)

m = Dropout(0.33)(m)



# m = Dense(1, activation='linear')(m)



output = Dense(y.shape[1], activation='softmax')(m)



model = Model(inputs=[input], outputs=[output])



model.summary()
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#配置训练模型，设置callback

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

#训练集上的loss不再减小时停止训练

es = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

#当标准评估停止提升时，降低学习速率

rlp = ReduceLROnPlateau(monitor='val_loss', patience=9, verbose=1, factor=0.5, cooldown=5, min_lr=1e-10)
#训练模型

history = model.fit(x_train

                    ,y_train

#                     ,validation_data=(x_test,y_test)

                    ,callbacks=[es, rlp]

                    ,verbose=1

                    ,epochs=30

                    , batch_size=512).history
import matplotlib.pyplot as plt

import seaborn as sns

#展示loss的收敛过程即准确率提升过程

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'], label='Train loss')

# ax1.plot(history['val_loss'], label='Validation loss')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['acc'], label='Train accuracy')

# ax2.plot(history['val_acc'], label='Validation accuracy')

ax2.legend(loc='best')

ax2.set_title('Accuracy')



plt.xlabel('Epochs')

sns.despine()

plt.show()
#用训练好的模型对测试集做出预测

model.evaluate(x_test, y_test)