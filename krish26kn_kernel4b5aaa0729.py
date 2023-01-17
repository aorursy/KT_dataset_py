#REQUIRED LIBRARIES

import numpy as np

import pandas as pd

from numpy.random import rand

import matplotlib.pyplot as plt

from datetime import datetime
#READING THE OIL PRICES DATA

data = pd.read_csv('../input/brent-oil-prices/BrentOilPrices.csv')
data.isnull()
print(data.info)
date = data['Date']
d = date[2000]

d[2]
for idx,d in enumerate(date):

    if (d[1]=='-') | (d[2]=='-'):

        date[idx] = datetime.strptime(date[idx],'%d-%b-%y')

    else:

        date[idx] = datetime.strptime(date[idx],'%b %d, %Y')

print(date)
#data['Date'] = pd.to_datetime(date, format="%d-%b-%y")

plt.plot(date,data['Price'])
price = data.Price

#price = price.iloc[:,:].values
price
def train_test_split(data,ratio):

    test_size = int(len(data)*0.3)

    train_size = len(data) - test_size

    train,test = data[0:train_size], data[train_size:len(data)]

    return train,test
train_data,test_data = train_test_split(price,0.2)
len(train_data)
def create_inputs(data,step_size=30,output_size=1):

    x = []

    y = []

    data = np.array(data)

    for i in range(len(data)):

        if (i+step_size+output_size)<len(data):

            x.append(data[i:i+step_size])

            y.append(data[i+step_size+output_size])

    return np.array(x),np.array(y)
class RNN:

    def __init__(self,input_size,output_size,hidden_size=64):

        self.Wx = rand(hidden_size,input_size)/10.0

        self.bx = rand(hidden_size,1)/10.0

        self.Wa = rand(hidden_size,hidden_size)/10.0

        self.Wy = rand(output_size,hidden_size)/10.0

        self.by = rand(output_size,1)/10.0

    def forward(self,inputs):

        a_prev = np.zeros((self.Wa.shape[0],1))

        self.a = {0:a_prev}

        for i,x in enumerate(inputs):

            a_prev = np.tanh(self.Wx.dot(x) + self.Wa.dot(a_prev) + self.bx)

            self.a[i+1] = a_prev

        yt = self.Wy.dot(a_prev) + self.by

        return yt

    def predict(self,inputs):

        a_prev = np.zeros((self.Wa.shape[0],1))

        for i,x in enumerate(inputs):

            a_prev = np.tanh(self.Wx.dot(x) + self.Wa.dot(a_prev) + self.bx)

        yt = self.Wy.dot(a_prev) + self.by

        return yt

    def backprop(self,dy,inputs,learn_rate=2e-2):

        n = len(inputs)

        a_last = self.a[n]

        dwy = dy.dot(a_last.T)

        dby = dy

        da = self.Wy.T.dot(dy)

        dwx = np.zeros(self.Wx.shape)

        dbx = np.zeros(self.bx.shape)

        dwa = np.zeros(self.Wa.shape)

        a0 = np.zeros((self.Wa.shape[0],1))

        for t in reversed(range(n)):

            temp = (1-(self.a[t+1])**2)*da

            dwx += temp.dot(inputs[t].T)

            dbx += temp

            dwa += temp.dot(self.a[t].T)

            #a0 += self.a[t+1]

            da = self.Wa.dot(temp)

        for d in [dwx,dwa,dbx,dwy,dby]:

            np.clip(d,-1,1,out=d)

        self.Wy -= learn_rate*dwy

        self.by -= learn_rate*dby

        self.Wx -= learn_rate*dwx

        self.bx -= learn_rate*dbx

        self.Wa -= learn_rate*dwa

        

            
rnn = RNN(1,1)
def train(data,e,learn_rate):

    x,y = create_inputs(data)

    err = 0

    for idx,x1 in enumerate(x):

        out = rnn.forward(x1)

        err += ((y[idx] - out)**2)/2

        dy = (out - y[idx])

        rnn.backprop(dy,x1,learn_rate)

    return err/x.shape[0]
for epochs in range(50):

    lr=0.01/1.2

    e=train(train_data,epochs,lr/(epochs+1))

    print(e,epochs)
x,y = create_inputs(test_data)

x.shape

y_t = []

for idx,x1 in enumerate(x):

        y_t.append(rnn.predict(x1))

y_t = np.array(y_t)
y_t = np.reshape(y_t,(y_t.shape[0],1))
plt.plot(y, color = 'red', label = 'Real Crude Oil Prices')

plt.plot(y_t, color = 'blue', label = 'Predicted Crude Oil Prices')

plt.title('Crude Oil Prices Prediction - MSE')

plt.xlabel('Time')

plt.ylabel('Crude Oil Prices')

plt.legend()

plt.show()