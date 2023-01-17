from customdatasets import TrainDataset, TestDataset
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import torch.optim as optim

from torchvision import datasets, transforms

from types import SimpleNamespace

import matplotlib.pyplot as plt

import csv

import librosa

import scipy as sc
# Hyperparameters

args = SimpleNamespace(batch_size=64, test_batch_size=64, epochs=3,

                       lr=0.01, momentum=0.5, seed=1, log_interval=200)

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')
import numpy as np



toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)



trainDataset = TrainDataset("../input/oeawai/train/kaggle-train", transform=toFloat)

print(len(trainDataset))



testDataset = TestDataset("../input/oeawai/kaggle-test/kaggle-test", transform=toFloat)

print(len(testDataset))



train_loader = torch.utils.data.DataLoader(trainDataset,

    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(testDataset,

        batch_size=args.test_batch_size, shuffle=False) #Shuffle should be false!
def logMagStft(numpyArray, sample_rate, n_fft):

    f, t, sx = sc.signal.stft(numpyArray, fs=sample_rate, nperseg=n_fft, noverlap=n_fft//2) 

    return np.log(np.abs(sx)+np.e**-10)



sample_rate = 16000

number_of_examples_to_plot = 8

n_fft = 510

spectrograms = np.zeros((number_of_examples_to_plot, n_fft//2+1, int(2*64000/n_fft)+2))

for samples, instrumentsFamily in train_loader:

    for index in range(number_of_examples_to_plot):

        spectrograms[index] = logMagStft(samples[index].numpy(), sample_rate, n_fft)

    family = trainDataset.transformInstrumentsFamilyToString(instrumentsFamily.numpy().astype(int))

    break # SVM is only fitted to a fixed size of data



import matplotlib.pyplot as plt

    

for i in range(number_of_examples_to_plot):

    print(spectrograms[i].shape)

    plt.imshow(spectrograms[i])

    print(family[i])

    plt.colorbar()

    plt.show()


import torch

from torch import nn



__author__ = 'Andres'





class LogConv(nn.Module):

    def __init__(self, log_size, in_channels, out_channels, kernel_size, stride, padding):

        super().__init__()

        self._log_size = log_size

        self._in_channels = in_channels

        self._out_channels = out_channels

        self._kernel_size = kernel_size

        self._stride = stride

        self._padding = padding



        self._ins, self._fins = self.ins_fins_for(self._log_size)

        self.logconv = nn.ModuleDict()



        for octave in range(len(self._ins)):

            self.logconv["octave_%d" % octave] = nn.utils.weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,

                                                   kernel_size=kernel_size, stride=stride, padding=padding))



    def split_sigs(self, x, ins, fins):

        xs = []

        for i, j in zip(ins, fins):

            xs.append(x[:, :, i:j, :])

        return xs



    def merge_sigs(self, x, ins, fins):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



        bs, ch, _, ts = x[0].shape

        m = torch.zeros([bs, ch, torch.max(torch.tensor(fins)), ts]).to(device)

        for d, i, j in zip(x, ins, fins):

            m[:, :, i:j, :] += d

        return m / 2



    def ins_fins_for(self, n=256):

        n = torch.tensor(n).long()

        ins = []

        fins = []

        ins.append(0)

        ins.append(0)

        fins.append(3)

        fins.append(6)



        for i in range(2, int(torch.log2(n.float())) + 1):

            iin = torch.tensor(2 ** i - 2 ** (i - 2))

            l = torch.tensor(2 ** (i + 1) + 2 ** (i - 2))

            ifin = iin + l

            ins.append(iin)

            fins.append(torch.min(ifin, n))

        return ins, fins



    def forward(self, x):

        results = []

        for octave, (ins, fins) in enumerate(zip(self._ins, self._fins)):

            results.append(self.logconv["octave_%d" % octave](x[:, :, ins:fins, :]))



        return self.merge_sigs(results, self._ins, self._fins)

# NN architecture (three conv and two fully connected layers)

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.first_conv = LogConv(256, 1, 10, 5, 1, 2)

        self.second_conv = LogConv(64, 10, 20, 5, 1, 2)

        self.third_conv = LogConv(16, 20, 50, 5, 1, 2)

        self.fc1 = nn.Linear(50*4*4, 500)

        self.fc2 = nn.Linear(500, 10)



    def forward(self, x):

        n_fft = 510

    

        spectrograms = np.zeros((len(x), n_fft//2+1, int(2*64000/n_fft)+2))

        for index, audio in enumerate(x.cpu().numpy()):

            spectrograms[index] = logMagStft(audio, 16000, n_fft)

        

        x = torch.from_numpy(spectrograms[:, np.newaxis, :, :]).to(device).float()

        x = nn.functional.pad(input=x, pad=(2, 2, 0, 0), mode='constant', value=0)

        # x.size is (batch_size, 1, 256, 252)

        x = F.relu(self.first_conv(x))

        x = F.max_pool2d(x, 4)

        x = F.relu(self.second_conv(x))

        x = F.max_pool2d(x, 4)

        x = F.relu(self.third_conv(x))

        x = F.max_pool2d(x, 4)

        # x.size is (batch_size, 50, 6, 6)

        x = x.view(-1, 4*4*50)



        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
# This function trains the model for one epoch

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))
# This function evaluates the model on the test data

def test(args, model, device, test_loader, epoch):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        familyPredictions = np.zeros(len(test_loader.dataset), dtype=np.int)

        for index, samples in enumerate(test_loader):

            samples = samples.to(device)

            familyPredictions[index*len(samples):(index+1)*len(samples)] = model(samples).max(1)[1].cpu() # get the index of the max log-probability

    

    familyPredictionStrings = trainDataset.transformInstrumentsFamilyToString(familyPredictions.astype(int))



    with open('NN-submission-' +str(epoch)+'.csv', 'w', newline='') as writeFile:

        fieldnames = ['Id', 'Predicted']

        writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',',

                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writeheader()

        for index in range(len(testDataset)):

            writer.writerow({'Id': index, 'Predicted': familyPredictionStrings[index]})

    print('saved predictions')
# Main

model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, 

                      momentum=args.momentum)



for epoch in range(1, args.epochs + 1):

    train(args, model, device, train_loader, optimizer, epoch)

    test(args, model, device, test_loader, epoch)
