import numpy as np

import pandas as pd
dataset = pd.read_csv("/kaggle/input/google-stock-price/Google_Stock_Price_Train.csv")
dataset.isnull().sum()
dataset.shape
dataset.head()
data = dataset.iloc[:, 1:2].values
print(data[:5])
data.shape
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(-1, 1))

data = scaler.fit_transform(data)

X_train = []

y_train = []

for i in range(50, 1258):

    X_train.append(data[i-50:i, 0])

    y_train.append(data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout
regressor = Sequential()

regressor.add(SimpleRNN(units=50,return_sequences = True,input_shape = (X_train.shape[1],1)))

regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50,return_sequences = True))

regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50,return_sequences = True))

regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50))

regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
regressor.summary()
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(X_train,y_train,epochs = 10, batch_size = 32)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_train.shape
import torch.nn as nn

import torch

from torch.autograd import Variable
INPUT_SIZE = 50

HIDDEN_SIZE = 40

NUM_LAYERS = 2

OUTPUT_SIZE = 1
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super(RNN, self).__init__()



        self.RNN = nn.RNN(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers

        )

        self.out = nn.Linear(hidden_size, output_size)



    def forward(self, x, h_state):

        r_out, hidden_state = self.RNN(x, h_state)

        

        hidden_size = hidden_state[-1].size(-1)

        r_out = r_out.view(-1, hidden_size)

        outs = self.out(r_out)



        return outs, hidden_state



RNN = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)



optimiser = torch.optim.Adam(RNN.parameters(), lr=0.01)

criterion = nn.MSELoss()



hidden_state = None



for epoch in range(100):

    inputs = Variable(torch.from_numpy(X_train).float())

    labels = Variable(torch.from_numpy(y_train).float())



    output, hidden_state = RNN(inputs, hidden_state) 



    loss = criterion(output.view(-1), labels)

    optimiser.zero_grad()

    loss.backward(retain_graph=True)                     # back propagation

    optimiser.step()                                     # update the parameters

    

    print('epoch {}, loss {}'.format(epoch,loss.item()))
