from sklearn import preprocessing

import pandas as pd

import numpy as np

def normalize_data(df):

    min_max_scaler = preprocessing.MinMaxScaler()

    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))

    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))

    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))

    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))

    #df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))

    return df



def data_split(stock, seq_len):

    amount_of_features = len(stock.columns) # 5

    data = stock.as_matrix()

    sequence_length = seq_len + 1 # index starting from 0

    result = []

    

    for index in range(len(data) - sequence_length + 1): # maxmimum date = lastest date - sequence length

        result.append(data[index: index + sequence_length]) # index : index + 15days

    

    result = np.array(result)

    row = round(0.85 * result.shape[0]) # 85% split

    train = result[:int(row), :] # 85% date, all features 

    

    x_train = train[:, :-1] 

    y_train =np.array(train[:, -1][:,1])

    

    x_test = result[int(row):, :-1] 

    y_test = np.array(result[int(row):, -1][:,1])



    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    

    return [x_train, y_train, x_test, y_test]
import torch

import torch.nn as nn

import math



# Here we define our model as a class

class test_LSTM(nn.Module):



    def __init__(self, input_dim, hidden_dim, num_layers=3,batch_size=1, output_dim=1):

        super(test_LSTM, self).__init__()

        self.input_dim = input_dim

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.num_layers = num_layers



        # Define the LSTM layer

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,batch_first =True)



        # Define the output layer

        self.linear = nn.Linear(self.hidden_dim, output_dim)



    def init_hidden(self):

        # This is what we'll initialise our hidden state as

        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),

                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))



    def forward(self, input):

        # Forward pass through LSTM layer

        # shape of lstm_out: [input_size, batch_size, hidden_dim]

        # shape of self.hidden: (a, b), where a and b both 

        # have shape (num_layers, batch_size, hidden_dim).

        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        

        # Only take the output from the final timetep

        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction

        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

        return y_pred.view(-1)
import torch

import torch.nn as nn

from torch.autograd import Variable

#from models import test_LSTM

from torch.utils.data import DataLoader

from torchvision import transforms

import pandas as pd

#from preprocess import data_split,normalize_data

import matplotlib.pyplot as plt

import numpy as np



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = '../input/NewYorkStockPrediction/'



df = pd.read_csv("../input/STT.csv", index_col = 0)



STT = df[df.symbol == 'STT'].copy()

#print(GOOG)

STT.drop(['symbol'],1,inplace=True)

STT.drop(['volume'],1,inplace=True)

STT_new = normalize_data(STT)

#print(GOOG_new)

window = 15

X_train, y_train, X_test, y_test = data_split(STT_new, window)



INPUT_SIZE = 4

HIDDEN_SIZE = 64

NUM_LAYERS = 1

OUTPUT_SIZE = 1



learning_rate = 0.001

num_epochs = 150



rnn = test_LSTM(input_dim=INPUT_SIZE,hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_dim=OUTPUT_SIZE)

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

criterion = nn.MSELoss()



rnn.to(device)

rnn.train()



for epoch in range(num_epochs):

    for inputs, label in zip(X_train,y_train):

        inputs = torch.from_numpy(inputs).float().to(device)

        label = torch.from_numpy(np.array(label)).float().to(device)

        optimizer.zero_grad()



        output =rnn(inputs) # forward   

        loss=criterion(output,label) # compute loss

        loss.backward() #back propagation

        optimizer.step() #update the parameters

    print('epoch {}, loss {}'.format(epoch,loss.item()))

# above for train



result = []

with torch.no_grad():

    for inputs, label in zip(X_test,y_test):

        inputs = torch.from_numpy(inputs).float().to(device)

        label = torch.from_numpy(np.array(label)).float().to(device)

        output =rnn(inputs)    

        result.append(output)

result =np.array(result)

# above for test



plt.figure(figsize=(20,5))

plt.plot(result,color='red', label='Prediction')

plt.plot(y_test,color='blue', label='Actual')

plt.legend(loc='best')

plt.show()

#print (X_train.shape, y_train.shape,X_test.shape,y_test.shape)

df = pd.read_csv("../input/STT.csv", index_col = 0)

STT = df[df.symbol == 'STT'].copy()

STT.drop(['symbol'],1,inplace=True)



min_max_scaler = preprocessing.MinMaxScaler().fit(STT['close'].values.reshape(-1,1))

denormalized_test = min_max_scaler.inverse_transform(y_test.reshape(-1,1))

denormalized_result = min_max_scaler.inverse_transform(result.reshape(-1,1))
from sklearn.metrics import mean_squared_error

mean_squared_error(denormalized_test, denormalized_result)*y_test.shape[0]