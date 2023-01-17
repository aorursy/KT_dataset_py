import pandas as pd

import numpy as np

import tensorflow as tf



import os

os.listdir()
summary=pd.read_csv('/kaggle/input/us-historical-stock-prices-with-earnings-data/dataset_summary.csv')

prices=pd.read_csv('/kaggle/input/us-historical-stock-prices-with-earnings-data/stocks_latest/stock_prices_latest.csv')

earnings=pd.read_csv('/kaggle/input/us-historical-stock-prices-with-earnings-data/stocks_latest/earnings_latest.csv')

dividends=pd.read_csv('/kaggle/input/us-historical-stock-prices-with-earnings-data/stocks_latest/dividends_latest.csv')
## prices

prices.head()
## earnings

earnings.head()
## dividends

dividends.head()
masterQuote=prices.merge(dividends,on=['symbol','date'],how='left')
masterQuote=masterQuote.fillna(0.0)

tickerCount=pd.DataFrame(masterQuote.groupby(['symbol']).count())

tickerCount=tickerCount.reset_index()

tickerCount=tickerCount[['symbol','date']]

qualified=tickerCount[tickerCount['date']>250]
msft=masterQuote[masterQuote['symbol']=='MSFT']

msft=msft.sort_values(by=['date'],ascending=True)
msft=msft[['close_adjusted','volume','split_coefficient','dividend']]
import matplotlib.pyplot as plt

histQuote=msft['close_adjusted'].tolist()

plt.plot(histQuote)

plt.show()
logRet=np.log(np.array(histQuote[1:])/np.array(histQuote[:-1]))
plt.plot(logRet)

plt.show()
paddedLogRet=[0]+list(logRet)
msft['logRet']=paddedLogRet
logvolume=np.log(msft['volume'].tolist())
msft['logVolume']=logvolume
def makeBatch():

    tp=msft

    trainXs=[]

    trainYs=[]

    tempTrain=[]

    for i in range(tp.shape[0]-1500-40):

        tempTrain=np.array(tp.iloc[i+0:i+40,2:6])

        trainXs.append(tempTrain)

        trainYs.append([tp['logRet'].tolist()[i+40]])

    return np.array(trainXs,dtype='float32'),np.array(trainYs,dtype='float32')

    
Xs,Ys=makeBatch()
## Last 1500 as test
tf.reset_default_graph()

num_hidden = 64



data = tf.placeholder(tf.float32, [None,40,4]) 

target = tf.placeholder(tf.float32, [None,1])

## LSTM

cell = tf.nn.rnn_cell.LSTMCell(num_hidden,activation='tanh',initializer=tf.random_normal_initializer())

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])

last = tf.gather(val, int(val.get_shape()[0]) - 1)

## Dense 1

weight = tf.Variable(tf.truncated_normal([num_hidden, 16]))

bias = tf.Variable(tf.constant(0.1, shape=[16]))

prediction1=tf.matmul(last, weight) + bias

## Dense 2

weight1 = tf.Variable(tf.truncated_normal([16,1]))

bias1 = tf.Variable(tf.constant(0.1, shape=[1]))

prediction = tf.matmul(prediction1, weight1) + bias1
val.shape
mse = tf.reduce_sum(tf.keras.losses.MSE(prediction,target))

optimizer = tf.train.AdamOptimizer(0.003).minimize(mse)

init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)

Xs,Ys=makeBatch()

batch_size=64

for i in range(200):

    ptr = 0

    for j in range(int(Xs.shape[0]/batch_size)-1):

        inp, out = Xs[j*batch_size:j*batch_size+batch_size], Ys[j*batch_size:j*batch_size+batch_size]

        train=sess.run(optimizer,{data: inp, target: out})

    loss=sess.run(mse,{data: inp, target: out})

    if (i+1)%100==0:

        print("Epoch - "+str(i)+": the loss is: " +str(loss))
predList=[]

actualList =[]



for j in range(int(Xs.shape[0]/batch_size)-1):

    inp, out = Xs[j*batch_size:j*batch_size+batch_size], Ys[j*batch_size:j*batch_size+batch_size]

    predTemp=sess.run(prediction,{data: inp})

    cellTemp=sess.run(val,{data:inp})

    actualList+=list(out)

    predList+=list(predTemp)
predReturn=[x[0] for x in predList]

actualReturn=[x[0] for x in actualList]
plt.plot(predReturn[:30],color='red')

plt.plot(actualReturn[:30],color='blue')

plt.show()
## Test

def makeBatchTest():

    tp=msft

    trainXs=[]

    trainYs=[]

    tempTrain=[]

    for i in range(tp.shape[0]-1500-40,tp.shape[0]-40-1):

        tempTrain=np.array(tp.iloc[i+0:i+40,2:6])

        trainXs.append(tempTrain)

        trainYs.append([tp['logRet'].tolist()[i+40]])

    return np.array(trainXs,dtype='float32'),np.array(trainYs,dtype='float32')
predList=[]

actualList =[]

Xs,Ys=makeBatchTest()

for j in range(int(Xs.shape[0]/batch_size)-1):

    inp, out = Xs[j*batch_size:j*batch_size+batch_size], Ys[j*batch_size:j*batch_size+batch_size]

    predTemp=sess.run(prediction,{data: inp})

    cellTemp=sess.run(val,{data:inp})

    actualList+=list(out)

    predList+=list(predTemp)
predReturn=[x[0] for x in predList]

actualReturn=[x[0] for x in actualList]

plt.plot(predReturn[:30],color='red')

plt.plot(actualReturn[:30],color='blue')

plt.show()
import math

actual_Price=msft['close_adjusted'].tolist()

predictChunk=actual_Price[-1500:]

predSeed=actual_Price[-1501]

generatedQuote=[]

for logret in predReturn:

    generatedQuote.append(predSeed*math.exp(logret))

    predSeed=predSeed*math.exp(logret)

plt.plot(generatedQuote)

plt.plot(predictChunk)

plt.show()
import tensorflow as tf

tf.reset_default_graph()

num_hidden = 64



data = tf.placeholder(tf.float32, [None,40,4]) 

target = tf.placeholder(tf.float32, [None,1])

## LSTM

cellfw = tf.nn.rnn_cell.LSTMCell(num_hidden,activation='tanh',initializer=tf.random_normal_initializer())

cellbw = tf.nn.rnn_cell.LSTMCell(num_hidden,activation='tanh',initializer=tf.random_normal_initializer())



lstm_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([cellfw])

lstm_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([cellbw])



valAll, state= tf.nn.bidirectional_dynamic_rnn(lstm_fw_multicell, lstm_bw_multicell, data, dtype=tf.float32)

out_fw, out_bw = valAll

output = tf.concat([out_fw, out_bw], axis=-1)

#val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(output , [1, 0, 2])

last = tf.gather(val, int(val.get_shape()[0]) - 1)

## Dense 1

weight = tf.Variable(tf.truncated_normal([num_hidden*2, 16]))

bias = tf.Variable(tf.constant(0.1, shape=[16]))

prediction1=tf.matmul(last, weight) + bias

## Dense 2

weight1 = tf.Variable(tf.truncated_normal([16,1]))

bias1 = tf.Variable(tf.constant(0.1, shape=[1]))

prediction = tf.matmul(prediction1, weight1) + bias1
mse = tf.reduce_sum(tf.keras.losses.MSE(prediction,target))

optimizer = tf.train.AdamOptimizer(0.003).minimize(mse)

init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)

Xs,Ys=makeBatch()

batch_size=64

for i in range(200):

    ptr = 0

    for j in range(int(Xs.shape[0]/batch_size)-1):

        inp, out = Xs[j*batch_size:j*batch_size+batch_size], Ys[j*batch_size:j*batch_size+batch_size]

        train=sess.run(optimizer,{data: inp, target: out})

    loss=sess.run(mse,{data: inp, target: out})

    if (i+1)%100==0:

        print("Epoch - "+str(i)+": the loss is: " +str(loss))
predList=[]

actualList =[]



for j in range(int(Xs.shape[0]/batch_size)-1):

    inp, out = Xs[j*batch_size:j*batch_size+batch_size], Ys[j*batch_size:j*batch_size+batch_size]

    predTemp=sess.run(prediction,{data: inp})

    cellTemp=sess.run(val,{data:inp})

    actualList+=list(out)

    predList+=list(predTemp)

predReturn=[x[0] for x in predList]

actualReturn=[x[0] for x in actualList]

plt.plot(predReturn[:30],color='red')

plt.plot(actualReturn[:30],color='blue')

plt.show()

## Test

def makeBatchTest():

    tp=msft

    trainXs=[]

    trainYs=[]

    tempTrain=[]

    for i in range(tp.shape[0]-1500-40,tp.shape[0]-40-1):

        tempTrain=np.array(tp.iloc[i+0:i+40,2:6])

        trainXs.append(tempTrain)

        trainYs.append([tp['logRet'].tolist()[i+40]])

    return np.array(trainXs,dtype='float32'),np.array(trainYs,dtype='float32')

predList=[]

actualList =[]

Xs,Ys=makeBatchTest()

for j in range(int(Xs.shape[0]/batch_size)-1):

    inp, out = Xs[j*batch_size:j*batch_size+batch_size], Ys[j*batch_size:j*batch_size+batch_size]

    predTemp=sess.run(prediction,{data: inp})

    cellTemp=sess.run(val,{data:inp})

    actualList+=list(out)

    predList+=list(predTemp)

predReturn=[x[0] for x in predList]

actualReturn=[x[0] for x in actualList]

plt.plot(predReturn[:30],color='red')

plt.plot(actualReturn[:30],color='blue')

plt.show()  
import math

actual_Price=msft['close_adjusted'].tolist()

predictChunk=actual_Price[-1500:]

predSeed=actual_Price[-1501]

generatedQuote=[]

for logret in predReturn:

    generatedQuote.append(predSeed*math.exp(logret))

    predSeed=predSeed*math.exp(logret)

plt.plot(generatedQuote)

plt.plot(predictChunk)

plt.show()