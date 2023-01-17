# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install yfinance 
!pip install pandas-datareader
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

# download dataframe using pandas_datareader
dataframe = pdr.get_data_yahoo("INFY")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(dataframe.Close, label='Infy')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price')
ax.legend()
no_of_features=4
scaler= MinMaxScaler(feature_range=(0, 1))
scaler_1 = MinMaxScaler(feature_range=(0, 1))
scaler_2 = MinMaxScaler(feature_range=(0, 1))
scaler_3 = MinMaxScaler(feature_range=(0, 1))
scaler_4 = MinMaxScaler(feature_range=(0, 1))

col_1=scaler_1.fit_transform(dataframe.Open.values.reshape(-1,1))
col_2=scaler_2.fit_transform(dataframe.High.values.reshape(-1,1))
col_3=scaler_3.fit_transform(dataframe.Low.values.reshape(-1,1))
col_4=scaler_4.fit_transform(dataframe.Close.values.reshape(-1,1))



data = scaler.fit_transform(dataframe[['Open', 'High', 'Low', 'Close']].values.reshape(-1,no_of_features)) # needs to change
# data = dataframe[['Open', 'High', 'Low', 'Close']].values.reshape(-1,no_of_features) # needs to change
data
import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_data(data, sequence_length, batch_size):
    data_x=[]
    data_y=[]
    for i,price in enumerate(data):
        if i+sequence_length<len(data):
            data_x.append(data[i:i+sequence_length])
            data_y.append(data[i+sequence_length,3])
        
    data = TensorDataset(torch.from_numpy(np.array(data_x)), torch.from_numpy(np.array(data_y)))

    dataloader = DataLoader(data, shuffle=False, batch_size=batch_size,drop_last=True)
    
    return dataloader

test_set_size = int(np.round(0.2*data.shape[0]))
validation_set_size = int(np.round(0.2*(data.shape[0] - (test_set_size))))
train_set_size = data.shape[0] -(validation_set_size)- (test_set_size)


x_train = data[:train_set_size,:]
x_validation = data[train_set_size:(train_set_size+validation_set_size),:]
x_test = data[(train_set_size+validation_set_size):,:]


sequence_length=10
batch_size_train=300
batch_size_validation=1
batch_size_test=1


train_loader=batch_data(x_train, sequence_length, batch_size_train)
validation_loader=batch_data(x_validation, sequence_length, batch_size_validation)
test_loader=batch_data(x_test, sequence_length, batch_size_test)

import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self,n_features,seq_length,dropout=0.2):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 3 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 dropout=dropout,
                                 batch_first = True)

        self.dropout = nn.Dropout(dropout)
        
        self.fc = torch.nn.Linear(self.n_hidden, 1)
        
    
    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        self.hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
    
    def forward(self, nn_input):        
        batch_size, seq_len, _ = nn_input.size()
        self.hidden = tuple([each.data for each in self.hidden])
        output, self.hidden = self.l_lstm(nn_input,self.hidden)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.n_hidden)
        output = self.fc(output)
        output = output.view(batch_size, -1)
        output = output[:, -1]
        return output 

# create NN
net= LSTM(no_of_features,sequence_length).cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

epochs = 25
training_losses=[]
validation_losses=[]
validation_output=[]
labels_output=[]
for t in tqdm(range(epochs)):
    
    #Training
    
    net.train()
    net.init_hidden(batch_size_train)
    loss_for_epoch=0
    i=0
    
    for batch_i, (inputs, labels) in enumerate(train_loader, 1):
        inputs = inputs.float().cuda()
        labels = labels.float().cuda()
        output = net(inputs) 
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()
        
        i+=1
        loss_for_epoch+=loss
    
#     print('step : ' , t , 'loss : ' , loss_for_epoch/i)
    training_losses.append(loss_for_epoch/i)
    
#     Validation
    
    net.eval()
    loss_for_epoch=0
    i=0
    net.init_hidden(batch_size_validation)
    for batch_i, (inputs, labels) in enumerate(validation_loader, 1):
        if t==epochs-1:
            labels_output.append(labels)
        inputs = inputs.float().cuda()
        labels = labels.float().cuda()
        output = net(inputs) 
        loss = criterion(output, labels)
                
        i+=1
        loss_for_epoch+=loss
        
        if t==epochs-1:
            validation_output.append(output.cpu().detach().numpy())
    
#     print('step : ' , t+1 , 'validation_loss : ' , loss_for_epoch/i)
    validation_losses.append(loss_for_epoch/i)
    if t==epochs-1:
        validation_output = np.concatenate(validation_output).ravel()
        
    
        

    
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(training_losses, label='Training_loss')
ax.plot(validation_losses, label='Validation_loss')

fig, ax = plt.subplots(figsize=(16,9))
ax.plot(validation_output, label='Validation_output')
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(labels_output, label='labels')
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(labels_output, label='labels')
ax.plot(validation_output, label='Validation_output')
net.eval()
labels_output=[]
predictions_output=[]
net.init_hidden(batch_size_validation)
for batch_i, (inputs, labels) in enumerate(test_loader, 1):
    
    labels_output.append(labels)
    inputs = inputs.float().cuda()
    labels = labels.float().cuda()
    output = net(inputs) 
    loss = criterion(output, labels)

    predictions_output.append(output.cpu().detach().numpy())

predictions_output = np.concatenate(predictions_output).ravel()
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(scaler_4.inverse_transform(predictions_output.reshape(-1, 1)), label='predictions_output')
print(len(np.array(labels_output).reshape(-1, 1)))
print(len(predictions_output.reshape(-1, 1)))
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(scaler_4.inverse_transform(np.array(labels_output).reshape(-1, 1)), label='labels')
ax.plot(scaler_4.inverse_transform(predictions_output.reshape(-1, 1)), label='predictions_output')
leg = ax.legend();