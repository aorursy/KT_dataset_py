# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
df_train_raw=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
df_train_raw=df_train_raw[df_train_raw['kills']<=30]
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import sklearn

df=df_train_raw.drop(columns=['Id','groupId','matchId','matchType'])

df=df.sample(n=100000)
df = df.values

df = np.array(df)

print(df)
for i in range(23):

    df[:,i] = (df[:,i]-df[:,i].min())/(df[:,i].max()-df[:,i].min())
x_data = df[:,:23]

y_data = df[:,23]
x = tf.placeholder(tf.float32,[None,23],name = "X")          #12个特征数据（12列）

y = tf.placeholder(tf.float32,[None,1],name = "Y")

Weights_L1 = tf.Variable(tf.random_normal([23,20],stddev=0.01))

biases_L1 = tf.Variable(tf.zeros([1,20]))

Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1

L1 = tf.tanh(Wx_plus_b_L1)

w = tf.Variable(tf.random_normal([20,1]))

b = tf.Variable(tf.zeros([1,20]))

Wx_plus_b_L2 = tf.matmul(L1,w) + b

pred = tf.nn.relu(Wx_plus_b_L2)
from random import shuffle

train_epochs = 1

learning_rate = 0.01

with tf.name_scope("LossFunction"):

    loss_function = tf.reduce_mean(tf.pow(y-pred,2))    #均方误差

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)



#声明会话

sess = tf.Session()



#定义初始化变量的操作

init = tf.global_variables_initializer()



#启动会话

sess.run(init)



#迭代训练

for epoch in range(train_epochs):

    loss_sum = 0.0

    for xs,ys in zip(x_data,y_data):

        

        xs = xs.reshape(1,23)

        ys = ys.reshape(1,1)

        #feed数据必须和Placeholder的shape一致

        #_,loss = sess.run(optimizer,feed_dict={x:xs,y:ys})

        _,loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})

        #loss_sum = loss_sum + loss

    #打乱数据顺序，防止按原次序假性训练输出

    #x_data,y_data = shuffle(x_data,y_data)

    

    b0temp = b.eval(session=sess)            #训练中当前变量b值

    w0temp = w.eval(session=sess)            #训练中当前权重w值

    loss_average = loss_sum/len(y_data)      #当前训练中的平均损失

    

    print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp,"w=",w0temp)

    #print("epoch=",epoch+1)
df_test1_raw=pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
df_test=df_test1_raw.drop(columns=['Id','groupId','matchId','matchType'])

df_test = df_test.values

df_test = np.array(df_test)

for i in range(23):

    df_test[:,i] = (df_test[:,i]-df_test[:,i].min())/(df_test[:,i].max()-df_test[:,i].min())

x__data_test = df_test[:,:23]

y__data_test = df_test[:,23]
row=len(df_test)

pre=np.zeros(row)

for n in range(1,row):

    x_test = x__data_test[n]

    x_test

    x_test = x_test.reshape(1,23)

    predict = sess.run(pred,feed_dict={x:x_test})

    #print(predict[:,1])

    #print("预测值：%f"%predict)

    pre[n]=predict[:,1]

print(pre)
df_submit=pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")
df_submit['winPlacePerc']=pre
print(df_submit)
df_submit.to_csv('../working/submission.csv',index=None)