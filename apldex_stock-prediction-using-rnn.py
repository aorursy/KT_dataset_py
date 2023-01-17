import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

%matplotlib inline

all_stock = pd.read_csv("../input/all_stocks_5yr.csv")

all_stock.head(10)
aapl = all_stock[all_stock.Name == 'AAPL']
aapl = aapl.close.values.astype('float32')
aapl = aapl.reshape(len(aapl), 1)

aapl.shape
fig=plt.figure(figsize=(18, 8), dpi= 80, facecolor='w', edgecolor='k')

scaler = MinMaxScaler(feature_range=(0,1))
aapl = scaler.fit_transform(aapl)

plt.plot(aapl)
plt.show()
train_size = int(len(aapl) * 0.80)
test_size = len(aapl) - train_size
train_set = aapl[0:train_size, :]
test_set = aapl[train_size:len(aapl), :]

print(len(train_set), len(test_set))
def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
        
    return np.array(dataX), np.array(dataY)
look_back = 1
trainX, trainY = create_dataset(train_set, look_back)
testX, testY = create_dataset(test_set, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainY = np.reshape(trainY, (trainY.shape[0], 1))
testY = np.reshape(testY, (testY.shape[0], 1))
testX.shape
input_size = 1
num_steps = 1
lstm_size = 128 # number of LSTM hidden unit
num_layers = 2 # number of LSTM cell
max_epoch = 200

inputs = tf.placeholder(tf.float32, [None, num_steps, input_size])
targets = tf.placeholder(tf.float32, [None, input_size])
learning_rate = tf.placeholder(tf.float32, None)
cell = tf.contrib.rnn.MultiRNNCell(
    [tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True) for _ in range(num_layers)], 
    state_is_tuple=True
) if num_layers > 1 else tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

weight = tf.Variable(tf.truncated_normal([lstm_size, input_size]))
bias = tf.Variable(tf.constant(0.1, shape=[input_size]))
prediction = tf.matmul(last, weight) + bias
loss = tf.reduce_mean(tf.square(prediction - targets))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
minimize = optimizer.minimize(loss)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_result = np.zeros((199, 1))
    test_result = np.zeros((50, 1))
    for epoch in range(max_epoch):
        train_loss, predicted, _ = sess.run([loss, prediction, minimize], 
                                            feed_dict={inputs: trainX, targets: trainY})
        if epoch % 20 == 0:
            print("Loss: ", train_loss)
        train_result = predicted
    predicted = sess.run([prediction], feed_dict={inputs: testX})
    test_result = predicted[0]
    
fig=plt.figure(figsize=(12, 8), dpi= 80, facecolor='w', edgecolor='k')

result = np.append(train_result, test_result)

result = np.reshape(result, (len(result), 1))

plt.plot(scaler.inverse_transform(aapl))
plt.plot(scaler.inverse_transform(result))
plt.legend(["Truth", "Predicted"], loc="upper left")
plt.show()

fig=plt.figure(figsize=(12, 8), dpi= 80, facecolor='w', edgecolor='k')

plt.plot(testY)
plt.plot(test_result)
plt.legend(["Target Truth", "Predicted"], loc="upper left")
plt.show()
for i in range(10):
    print("Target: ", testY[i], "Predicted: ", test_result[i])
