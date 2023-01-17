# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/train"))



import pandas as pd

import matplotlib.pyplot as plt

import IPython.display as ipd



#import mock

#import sys



import librosa

import librosa.display

import torchaudio

import torch.nn as nn

import torch

from torch.utils import data

# Any results you write to the current directory are saved as output.

# First checking if GPU is available

train_on_gpu=torch.cuda.is_available()



if(train_on_gpu):

    print('Training on GPU.')

else:

    print('No GPU available, training on CPU.')
Labels = pd.read_csv("../input/train/train.csv")

WavPath = "../input/train/Train/"

Fils = os.listdir(WavPath)

sound, sample_rate = torchaudio.load(WavPath+Fils[17])

ipd.Audio(data=sound[1,:],rate=sample_rate) # load a local WAV file



plt.figure(figsize=(20, 5))

plt.scatter(range(100),sound[1,1000:1100])

plt.scatter(range(0,100,5),sound[1,1000:1100:5])
x, sr = librosa.load(WavPath+Fils[17])



plt.figure(figsize=(14, 5))

librosa.display.waveplot(x, sr=sr)

X = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

FilesS = np.zeros(len(Fils))

for i,F in enumerate(Fils):

    FilesS[i] = os.path.getsize(WavPath+F)



plt.figure(figsize=(20,8))

plt.hist(FilesS,bins=50)



# Encode classes

ClassDict = dict(enumerate(set(Labels['Class'])))

Class2int = {ch: ii for ii, ch in ClassDict.items()}

encoded = np.array([Class2int[ch] for ch in Labels['Class']])





## split data into training, validation, and test data (features and labels, x and y)

split_frac = 0.8

batch_size = 8



split_idx = int(len(Fils)*split_frac)

split_idx1 = int(batch_size*np.floor(split_idx/batch_size))

split_idx2 = int(batch_size*np.floor( (len(Fils) - split_idx1)/batch_size ))

train_x, val_x = Fils[:split_idx1], Fils[split_idx1:split_idx1+split_idx2]

train_y, val_y = encoded[:split_idx1], encoded[split_idx1:split_idx1+split_idx2]

print(len(train_x)/batch_size, len(val_x)/batch_size )


class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels,DataPath,RecLen,DecNum=5):

        'Initialization'

        self.labels = labels

        self.list_IDs = list_IDs

        self.DataPath = DataPath

        #self.RecLen = 176400 # length of most records

        self.RecLen = RecLen # length of most records

        self.DecNum = DecNum

    def __len__(self):

        'Denotes the total number of samples'

        return len(self.list_IDs)



    def __getitem__(self, index):

        'Generates one sample of data'

        # Select sample

        ID = self.list_IDs[index]



        # Load data and get label

        X ,_= torchaudio.load(self.DataPath + ID  )

        while X.shape[1]<self.RecLen :  # duplicate short recorts

            X = torch.cat((X,X),dim=1)

        X = X[0,0:self.RecLen:self.DecNum] # cut records to have the same length

        y = self.labels[ID]



        return X, y
class SOUND_RNN(nn.Module):

    

    def __init__(self, Data_x, n_hidden=256, n_layers=2,

                               drop_prob1=0.5,drop_prob2=0.25, lr=0.001):

        super().__init__()

        self.drop_prob1 = drop_prob1

        self.drop_prob2 = drop_prob2

        self.n_layers = n_layers

        self.n_hidden = n_hidden

        self.lr = lr

        

        self.Data_x = Data_x

                

        #self.lstm = nn.LSTM(len(self.Data_x),hidden_size=n_hidden,

        self.lstm = nn.LSTM(1,hidden_size=n_hidden,

                            num_layers = n_layers,dropout= drop_prob1, batch_first=True,bidirectional=False)

        self.rnn = nn.RNN(1,hidden_size=n_hidden,

                            num_layers = n_layers,dropout= drop_prob1, batch_first=True,bidirectional=False)

        self.gru = nn.GRU(1,hidden_size=n_hidden,

                            num_layers = n_layers,dropout= drop_prob1, batch_first=True,bidirectional=True)

        self.Drop = nn.Dropout(drop_prob2)

        self.fc1 = nn.Linear(2*n_hidden,10)

        self.fc2 = nn.Linear(64,10)

      

    

    def forward(self, x, hidden):

        ''' Forward pass through the network. 

            These inputs are x, and the hidden/cell state `hidden`. '''

        batch_size = x.size(0)

        x = x.unsqueeze_(2)

        #x,hidden = self.lstm(x,hidden)

        x,hidden = self.gru(x,hidden)

        out = self.Drop(x)

        out = out[:,-1,:]

        #out = out.contiguous().view(-1, self.n_hidden)

        out = self.fc1(self.Drop(out))

        out = out.view(batch_size, -1)



        # return the final output and the hidden state

        return out, hidden

    

    

    def init_hidden(self, batch_size,isNotTuple = False):

        ''' Initializes hidden state '''

        # Create two new tensors with sizes n_layers x batch_size x n_hidden,

        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        

        if (train_on_gpu):

            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),

                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())

            if(isNotTuple):

                 hidden = weight.new(2*self.n_layers, batch_size, self.n_hidden).zero_().cuda()

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),

                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        

        

        return hidden
def one_hot_encode(arr, n_labels):

    

    # Initialize the the encoded array

    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    

    # Fill the appropriate elements with ones

    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    

    # Finally reshape it to get back to the original array

    one_hot = one_hot.reshape((*arr.shape, n_labels))

    

    return one_hot



one_hot = one_hot_encode(encoded.reshape(-1,1), 10)

one_hot_train = one_hot_encode(train_y.reshape(-1,1), 10)

one_hot_val = one_hot_encode(val_y.reshape(-1,1), 10)



labelsDict = dict(zip(Fils,one_hot))

labelsDict_train = dict(zip(train_x,one_hot_train))

labelsDict_val = dict(zip(val_x,one_hot_val))



params = {'batch_size': batch_size,

          'shuffle': True,

          'num_workers': 2}



RecLen = 56400

training_set = Dataset(train_x, labelsDict_train,WavPath,RecLen)

training_generator = data.DataLoader(training_set, **params)



val_set = Dataset(val_x, labelsDict_val,WavPath,RecLen)

val_generator = data.DataLoader(val_set, **params)





n_hidden= 64

n_layers= 2

net = SOUND_RNN(np.zeros((batch_size,RecLen)), n_hidden, n_layers)



clip = 5

if(train_on_gpu):

    net.cuda()



opt = torch.optim.Adam(net.parameters(), lr=0.003)

criterion = nn.CrossEntropyLoss()



max_epochs = 3

counter = 0 

print_every = 20

print("Start training")

for epoch in range(max_epochs):



    h = net.init_hidden(batch_size,isNotTuple=True)

    for local_batch, local_labels in training_generator:

        counter += 1

        net.train()



        if(train_on_gpu):

            local_batch, local_labels = local_batch.cuda(), local_labels.cuda()



        #h = tuple([each.data for each in h])

        

        # zero accumulated gradients

        net.zero_grad()



        # get the output from the model

        output, h = net(local_batch, h)



        # calculate the loss and perform backprop

        loss = criterion(output,torch.squeeze(torch.argmax(local_labels,dim=-1)))

        loss.backward(retain_graph=True)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

        nn.utils.clip_grad_norm_(net.parameters(), clip)

        opt.step()

        

        if counter % print_every == 0:

            # Get validation loss

            val_h = net.init_hidden(batch_size,isNotTuple=True)

            val_losses = []

            net.eval()

            for inputs, labels in val_generator:



                # Creating new variables for the hidden state, otherwise

                # we'd backprop through the entire training history

                #val_h = tuple([each.data for each in val_h])



                if(train_on_gpu):

                    inputs, labels = inputs.cuda(), labels.cuda()



                output, val_h = net(inputs, val_h)

                val_loss = criterion(output.squeeze(), torch.squeeze(torch.argmax(labels,dim=-1)))



                val_losses.append(val_loss.item())



            net.train()

            print("Epoch: {}/{}...".format(epoch+1, max_epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)))




