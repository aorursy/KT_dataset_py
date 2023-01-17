#import packages

import torch 

import torch.nn as nn

from torch.autograd import Variable

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
#load and print data

data = pd.read_csv("../input/pmsm_temperature_data.csv")

data.head(10)
#no need to standardize data, std nearly 1, mean nearly 0

data.describe()
#no missing values

print(data.shape)

data.info()
#overview about sessions and length

sns.catplot(x="profile_id", kind="count", data=data, height=9, aspect=16/9, palette=sns.color_palette(['blue']))

plt.show()
#function to trim sessions to an appropriate size

def preprocess(sessions_id, seq_len, target="torque"):

    sessions = []

    for id in sessions_id:

        s = data[data["profile_id"] == id]

        r = len(s) % seq_len

        l = len(s) - r

        

        session = s.iloc[:l]

        

        y = session[target]

        X = session.drop([target, "profile_id"], axis=1)

        

        X = torch.from_numpy(X.values).float() 

        y = torch.from_numpy(y.values).float()



        sessions.append((X, y))

        

    return sessions
#Bidirectional LSTM class

class NET(nn.Module):

    def __init__(self, in_size, h_size, n_layers, out_size):

        super(NET, self).__init__()

        self.h_size = h_size

        self.n_layers = n_layers

        self.out_size = out_size

        self.lstm = nn.LSTM(in_size, h_size, n_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(h_size * 2, out_size)

             

    def forward(self, x):

        #init. states 

        h0 = torch.zeros(self.n_layers * 2, x.size(0), self.h_size)

        c0 = torch.zeros(self.n_layers * 2, x.size(0), self.h_size)

        

        out, _ = self.lstm(x, (h0, c0)) 

        out = self.fc(out)

        return out 
#defining the sequence length and the sessions ids which are used for training

seq_len = 100

ids = [4, 6, 10, 11, 20, 27, 29, 30, 31,32, 36] + [i+41 for i in range(35)]



#trim sessions

sessions = preprocess(ids, seq_len)



#create an lstm instance

model = NET(11, 20, 2, 1)

print(model)



#define optimizer and loss function

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

criterion = nn.MSELoss()
n_epochs = 30



#train model

train_losses = []



model.train()

for e in range(n_epochs):

    for X, y in sessions:

        optimizer.zero_grad()

        out = model.forward(X.view(-1, seq_len, 11))

        loss = criterion(out, y.view(-1, seq_len, 1))

        

        loss.backward()

        optimizer.step()

        train_losses.append(loss.detach().numpy())

    

#print training error

plt.rcParams["figure.figsize"] = [16, 9]

plt.plot(train_losses, 'b', label='Training Error')

plt.legend(loc='upper right')

plt.ylabel('Error')

plt.show()
model.eval()



#evaluation on session 80 and 81

session_80 = data[data["profile_id"] == 80]

session_80 = session_80.iloc[:15500]



y = session_80["torque"]

X = session_80.drop(["torque", "profile_id"], axis=1)



X_80 = torch.from_numpy(X.values[:22000, :]).float() 

y_80 = torch.from_numpy(y.values[:22000]).float()



out = model.forward(X_80.view(-1, seq_len, 11))

loss = criterion(out, y_80.view(-1, seq_len, 1))

print("Loss: " + str(loss.detach().numpy()))



out = out[:, -1, :]

pred = out.detach().numpy().flatten()

real = y_80.numpy()[seq_len::seq_len]



plt.plot(real.flatten(), label="real values")

plt.plot(pred, label="predicted values")

plt.legend(loc='upper right')

plt.show()
#evaluation on session 81

session_81 = data[data["profile_id"] == 81]

session_81 = session_81.iloc[:15000]



y = session_81["torque"]

X = session_81.drop(["torque", "profile_id"], axis=1)



X_81 = torch.from_numpy(X.values[:15000, :]).float() 

y_81 = torch.from_numpy(y.values[:15000]).float()



out = model.forward(X_81.view(-1, seq_len, 11))

loss = criterion(out, y_81.view(-1, seq_len, 1))

print("Loss: " + str(loss.detach().numpy()))



out = out[:, -1, :]

pred = out.detach().numpy().flatten()

real = y_81.numpy()[seq_len::seq_len]



plt.plot(real.flatten(), label="real values")

plt.plot(pred, label="predicted values")

plt.legend(loc='upper right')

plt.show()