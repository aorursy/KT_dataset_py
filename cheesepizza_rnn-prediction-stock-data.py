import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
xy = np.loadtxt('../input/stock_daily.csv',dtype=np.float32,delimiter=',')
"""
매일 최신순으로 정렬된 주식 데이터를 가지고 7일치씩 끊어서 train시키고 
8일차의 close값을 예측하여라
""" 
xy=xy[::-1] #사람은 배워야해
# dc = data[:,:] 
# for i in range(int(len(data)/2)):
#     dc[i] = data[-i]
#     dc[-i] = data[i]
def MinMaxScalar(data):
    bunja = data-np.min(data,0)
    bunmo = np.max(data,0) - np.min(data,0)
    return bunja/(bunmo+1e-10)
xy= MinMaxScalar(xy)
data_dim = 5
hidden_dim = 16
output_dim = 1 #결과값은 다음날 close값 하나임
seq_length = 7
learning_rate = 0.1
x = xy
y = xy[:,[-1]]
np.array(y).shape
dataX = []
dataY = []
for i in range(0, len(y)-seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    dataX.append(_x)
    dataY.append(_y)
# 둘다 같은거
batch_size = len(dataX)
batch_size = len(xy)-seq_length
train_size = int(len(dataY)*0.7)
test_size = len(dataY)- train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])
testX.shape
X = tf.placeholder(tf.float32, shape=[None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, shape=[None,1])
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True,
                                  activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell,X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim,
                                         activation_fn=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(Y_pred-Y)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

target = tf.placeholder(tf.float32, [None,1])
prediction = tf.placeholder(tf.float32, [None,1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(target-prediction)))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2000):
   l,_ = sess.run([loss, train], feed_dict={X:dataX, Y:dataY})
   if step % 100 == 0:
       print ("step:",step,"loss:",l)
test_prediction = sess.run(Y_pred, feed_dict={X:testX})
print ("RMSE:",sess.run(rmse, feed_dict={target:testY, prediction:test_prediction}))
plt.plot(testY)
plt.plot(test_prediction)






