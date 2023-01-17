import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from librosa import feature

import librosa 

import matplotlib.pyplot as plt

import IPython.display as ipd  # To play sound in the notebook

import librosa.display

import json

from matplotlib.pyplot import specgram

import pandas as pd

import seaborn as sns

import glob 

import os

from tqdm import tqdm

import pickle

from PIL import Image



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn import init

from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision import transforms, utils

from torch.autograd import Variable

from torchvision import models
# Use one audio file in previous parts again

import os

MUSIC = '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original'

music_dataset = []

genre_target = []

for root, dirs, files in os.walk(MUSIC):

    for name in files:

        filename = os.path.join(root, name)

        if filename != '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/jazz/jazz.00054.wav':

            music_dataset.append(filename)

            genre_target.append(filename.split("/")[6])
music_dataset[67]
mel_spec=[]

genre_new=[]

N_FFT = 512

N_MELS = 96

HOP_LEN = 256

num_div=8

for idx, wav in enumerate(music_dataset):

    y, sfr = librosa.load(wav)

    div= np.split(y[:660000], num_div)

    for chunck in div:

        melSpec = librosa.feature.melspectrogram(y=chunck, sr=sfr, n_mels=N_MELS,hop_length=HOP_LEN, n_fft=N_FFT)

        melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)

        mel_spec.append(melSpec_dB)

        genre_new.append(genre_target[idx])
genres={'pop':1,'classical':2,'reggae':3,'disco':4,'jazz':5,'metal':6,'country':7,'blues':8,'hiphop':9,'rock':0}

genre_id = [genres[item] for item in genre_new]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(mel_spec, genre_id, test_size=0.2, random_state=42)
BATCH_SIZE = 256



torch_X_train = torch.unsqueeze(torch.cuda.FloatTensor(X_train),1)

torch_y_train = torch.cuda.LongTensor(y_train)



# create feature and targets tensor for test set.

torch_X_test = torch.unsqueeze(torch.cuda.FloatTensor(X_test),1)

torch_y_test = torch.cuda.LongTensor(y_test)



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)

test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
class upchannel(nn.Module):

    def __init__(self):

        super(upchannel, self).__init__()



        self._convblocks = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),



            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),



            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=4),



            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(512),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=4)

        )

        self._classifier = nn.Sequential(nn.Linear(in_features=512*5, out_features=1024),

                                         nn.ReLU(),

                                         nn.Dropout(),

                                         nn.Linear(in_features=1024, out_features=256),

                                         nn.ReLU(),

                                         nn.Dropout(),

                                         nn.Linear(in_features=256, out_features=10))

        self.apply(self._init_weights)



    def forward(self, x):

        x = self._convblocks(x)

        x = x.view(x.size(0), -1)

        score = self._classifier(x)

        return score



    def _init_weights(self, layer) -> None:

        if isinstance(layer, nn.Conv1d):

            nn.init.kaiming_uniform_(layer.weight)

        elif isinstance(layer, nn.Linear):

            nn.init.xavier_uniform_(layer.weight)
model=upchannel()

model.cuda()

error = nn.CrossEntropyLoss()

learning_rate=0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
EPOCHS = 60

model.train()

for epoch in range(EPOCHS):

    correct = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

        var_X_batch = Variable(X_batch).float()

        var_y_batch = Variable(y_batch)

        optimizer.zero_grad()

        output = model(var_X_batch)

        loss = error(output, var_y_batch)

        loss.backward()

        optimizer.step()



                # Total correct predictions

        predicted = torch.max(output.data, 1)[1] 

        correct += (predicted == var_y_batch).sum()

                #print(correct)

        if batch_idx % 50 == 0:

            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
torch.save(model.state_dict(),'./CNN60.pth')
model.eval()
correct = 0

total = 0

with torch.no_grad():

    for data in test_loader:

        images, labels = data

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the test images: %d %%' % (

    100 * correct / total))
BATCH_SIZE = 128



torch_X_train = torch.unsqueeze(torch.cuda.FloatTensor(X_train),1)

torch_y_train = torch.cuda.LongTensor(y_train)



# create feature and targets tensor for test set.

torch_X_test = torch.unsqueeze(torch.cuda.FloatTensor(X_test),1)

torch_y_test = torch.cuda.LongTensor(y_test)



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)

test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
import torch

torch.manual_seed(123)

import torch.nn as nn



class CRNN(nn.Module):

    def __init__(self):

        super(CRNN, self).__init__()

        

        self._extractor = nn.Sequential(

             nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),

             nn.BatchNorm2d(64),

             nn.ReLU(),

             nn.MaxPool2d(kernel_size=2),

 

             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),

             

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),



            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),

                         nn.BatchNorm2d(512),

                         nn.ReLU(),

                         nn.MaxPool2d(kernel_size=4)

        )

        self._rnnModule = nn.Sequential(

            nn.GRU(512*3, 512, batch_first=False,bidirectional=True)

            #nn.LSTM(512, 512, batch_first=False, bidirectional=True),

        )

        

        self._classifier = nn.Sequential(nn.Linear(in_features=10*1024, out_features=512),

                                         nn.ReLU(),

                                         nn.Dropout(),

                                         nn.Linear(in_features=512, out_features=256),

                                         nn.ReLU(),

                                         nn.Dropout(),

                                         nn.Linear(in_features=256, out_features=10))

        self.apply(self._init_weights)

    def forward(self, x):

        x = self._extractor(x)

        x = x.permute(3,0,1,2)

        x = x.view(x.size(0), x.size(1), -1)

        x, hn = self._rnnModule(x)

        x = x.permute(1, 2, 0)

        #print(x.shape)

        x = x.reshape(x.size(0), -1)

        score = self._classifier(x)

        return score

    def _init_weights(self, layer) -> None:

        if isinstance(layer, nn.Conv1d):

            nn.init.kaiming_uniform_(layer.weight)

        elif isinstance(layer, nn.Linear):

            nn.init.xavier_uniform_(layer.weight)
model=CRNN()

model.cuda()

error = nn.CrossEntropyLoss()

learning_rate=0.0005

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
EPOCHS = 60

model.train()

for epoch in range(EPOCHS):

    correct = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

        var_X_batch = Variable(X_batch).float()

        var_y_batch = Variable(y_batch)

        optimizer.zero_grad()

        output = model(var_X_batch)

        loss = error(output, var_y_batch)

        loss.backward()

        optimizer.step()



                # Total correct predictions

        predicted = torch.max(output.data, 1)[1] 

        correct += (predicted == var_y_batch).sum()

                #print(correct)

        if batch_idx % 50 == 0:

            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
model.eval()
correct = 0

total = 0

with torch.no_grad():

    for data in test_loader:

        images, labels = data

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the test images: %d %%' % (

    100 * correct / total))
torch.save(model.state_dict(),'./CRNN60.pth')
test,sfr = librosa.load('/kaggle/input/song-test/Beyonc - Crazy In Love ft (mp3cut.net).wav')

testspec = librosa.feature.melspectrogram(y=chunck, sr=sfr, n_mels=N_MELS,hop_length=HOP_LEN, n_fft=N_FFT)

testspec = librosa.power_to_db(testspec, ref=np.max)