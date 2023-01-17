# code from https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html



# version 2, edited in a way, that no test data exist, we just predict continouisly



# version 3, what if we train with different frequencies, is the network able to generalize?



# version 4, prediction worked well for frequency that existed in training data, but just as long as input is

#            taken constatnly from dataset, using the predicted values one gets an other frequency, 

#            test now for a intermediate frequency



# version 5, lets try that the network has to predict the next k steps, to get more stable training 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, SimpleRNN



N = 10000        



t=np.arange(0,N)

#uniform noise between 0 and 1

f=[0.005, 0.01, 0.015, 0.02]

x=[]

for f_ in f:

    x= np.append(x, np.sin(f_*t)+1*np.random.rand(N))

    

df = pd.DataFrame(x)

df.head()



plt.plot(df)

plt.show() 



values=df.values

train = values[:,:]



f_test = 0.0125

test = np.sin(f_test*t)+1*np.random.rand(N)

test = np.reshape(test, (test.size, 1))



print("train: {}, test: {}".format(train.shape, test.shape))
step = 1100

k = 200
# convert into dataset matrix

def convertToMatrix(data, step):

    X, Y =[], []

    for i in range(len(data)-step-k):

        d=i+step  

        X.append(data[i:d,])

        Y.append(data[d:d+k,])

    return np.array(X), np.array(Y)



trainX,trainY =convertToMatrix(train,step)

testX,testY =convertToMatrix(test,step)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1]))



print(trainX.shape)

print(testX.shape)
# SimpleRNN model

model = Sequential()

model.add(SimpleRNN(units=16*k, input_shape=(1,step), activation="relu"))

model.add(Dense(4*k, activation="relu")) 

model.add(Dense(k))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.summary()
model.fit(trainX,trainY, epochs=100, batch_size=16, verbose=2)
testXs = testX[0:2*step+1,0,:]

testXs = np.reshape(testXs, (2*step+1, 1, step))

print(testXs.shape)



predicted = model.predict(testXs)

print("prdicted shape: {}".format(predicted.shape))



predicted = predicted[:,0]

predicted = np.reshape(predicted, (len(predicted)))



print(predicted.shape)

#using the testX set would be kind of cheating, lets just use the predicted values to continue

print("model fited, start prediction")

for i in range(0, len(testX)-2*step):

    past_window = predicted[-step-1:-1]

    #print("window length is {}".format(len(past_window)))

    np_window = np.reshape(past_window, (1, 1, step))

    

    p = model.predict(np_window)

    p = np.reshape(p[0,0], (1))

    predicted=np.concatenate((predicted,p),axis=0)



print("number of predicted signals is {}".format(predicted.shape))
#important correction to original, prediction should be shifted by +step (for the index)!

plt.plot(np.arange(0,N),test)

plt.plot(np.arange(0,N-step+1-k)+step,predicted)

plt.axvline(3*step+1, c="r")

plt.show() 