import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

import torch.nn.functional as F

import os



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class BasicRNN(nn.Module):

    def __init__(self, n_inputs, n_neurons):    

        super(BasicRNN, self).__init__()

        

        self.Wx = torch.randn(n_inputs, n_neurons) # n_inputs X n_neurons

        self.Wy = torch.randn(n_neurons, n_neurons) # n_neurons X n_neurons

        self.b = torch.zeros(1, n_neurons) # 1 X n_neurons

        

    def forward(self, X0, X1):

        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b) # batch_size X n_neurons        

        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) + torch.mm(X1, self.Wx) + self.b) # batch_size X n_neurons

        return self.Y0, self.Y1    
N_INPUT = 3 # number of features in input

N_NEURONS = 5 # number of units in layer



X0_batch = torch.tensor([[0,1,2], [3,4,5], 

                         [6,7,8], [9,0,1]],

                        dtype = torch.float) #t=0 => 4 X 3



X1_batch = torch.tensor([[9,8,7], [0,0,0], 

                         [6,5,4], [3,2,1]],

                        dtype = torch.float) #t=1 => 4 X 3



model = BasicRNN(N_INPUT, N_NEURONS)



Y0_val, Y1_val = model(X0_batch, X1_batch)
print(Y0_val)

print(Y1_val)
rnn = nn.RNNCell(3, 5) # n_input X n_neurons

X_batch = torch.tensor([[[0,1,2], [3,4,5], 

                         [6,7,8], [9,0,1]],

                        [[9,8,7], [0,0,0], 

                         [6,5,4], [3,2,1]]

                       ], dtype = torch.float) # X0 and X1

hx = torch.randn(4, 5) # m X n_neurons

output = []



# for each time step

for i in range(2):

    hx = rnn(X_batch[i], hx)

    output.append(hx)



print(output)
class CleanBasicRNN(nn.Module):

    def __init__(self, batch_size, n_inputs, n_neurons):

        super(CleanBasicRNN, self).__init__()

        

        rnn = nn.RNNCell(n_inputs, n_neurons)

        self.hx = torch.randn(batch_size, n_neurons) # initialize hidden state

        

    def forward(self, X):

        output = []



        # for each time step

        for i in range(2):

            self.hx = rnn(X[i], self.hx)

            output.append(self.hx)

        

        return output, self.hx



FIXED_BATCH_SIZE = 4 # our batch size is fixed for now

N_INPUT = 3

N_NEURONS = 5



X_batch = torch.tensor([[[0,1,2], [3,4,5], 

                         [6,7,8], [9,0,1]],

                        [[9,8,7], [0,0,0], 

                         [6,5,4], [3,2,1]]

                       ], dtype = torch.float) # X0 and X1





model = CleanBasicRNN(FIXED_BATCH_SIZE, N_INPUT, N_NEURONS)

output_val, states_val = model(X_batch)

print(output_val) # contains all output for all timesteps

print(states_val) # contains values for final state or final timestep, i.e., t=1