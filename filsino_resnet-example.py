import os

import glob

import numpy as np

import re

import scipy.io.wavfile

import torch

import torch.utils.data as data

import torchvision.transforms as transforms

from sklearn.preprocessing import LabelEncoder





class TrainDataset(data.Dataset):

    """Pytorch dataset for instruments

    args:

        root: root dir containing an audio directory with wav files.

        transform (callable, optional): A function/transform that takes in

                a sample and returns a transformed version.

        blacklist_patterns: list of string used to blacklist dataset element.

            If one of the string is present in the audio filename, this sample

            together with its metadata is removed from the dataset.

    """



    def __init__(self, filenames, transform=None, blacklist_patterns=[]):

        assert(isinstance(root, str))

        assert(isinstance(blacklist_patterns, list))



        self.filenames = filenames 



        for pattern in blacklist_patterns:

            self.filenames = self.blacklist(self.filenames, pattern)

            

        self.labelEncoder = LabelEncoder() # Encode labels with value between 0 and n_classes-1.

        self.labelEncoder.fit(np.unique(self._instrumentsFamily(self.filenames)))

            

        self.transform = transform

        

    def transformInstrumentsFamilyToString(self, targets=[]):

        return self.labelEncoder.inverse_transform(targets) # Decode values into labels

                    

    def _instrumentsFamily(self, filenames):

        instruments = np.zeros(len(filenames), dtype=object)

        for i, file_name in enumerate(filenames): # Extract family name from filename

            no_folders = re.compile('\/').split(file_name)[-1]

            instruments[i] = re.compile('_').split(no_folders)[0]

        return instruments

    

    def blacklist(self, filenames, pattern):

        return [filename for filename in filenames if pattern not in filename]



    def __len__(self):

        return len(self.filenames)



    def __getitem__(self, index):

        name = self.filenames[index]

        _, sample = scipy.io.wavfile.read(name) # load audio

        

        target = self._instrumentsFamily([name])

        categorical_target = self.labelEncoder.transform(target)[0]

                

        if self.transform is not None:

            sample = self.transform(sample)

        return [sample, categorical_target]
from customdatasets import TestDataset
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

args = SimpleNamespace(batch_size=64, test_batch_size=64, epochs=10,

                       lr=0.01, momentum=0.5, seed=1, log_interval=2000)

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')
import numpy as np



toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)



root = "../input/oeawai/train/kaggle-train"

filenames = glob.glob(os.path.join(root, "audio/*.wav"))

np.random.shuffle(filenames)



trainDataset = TrainDataset(filenames[:int(len(filenames)*0.95)], transform=toFloat)

print(len(trainDataset))

validDataset = TrainDataset(filenames[int(len(filenames)*0.95):], transform=toFloat)

print(len(validDataset))



testDataset = TestDataset("../input/oeawai/kaggle-test/kaggle-test", transform=toFloat)

print(len(testDataset))



train_loader = torch.utils.data.DataLoader(trainDataset,

    batch_size=args.batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(validDataset,

    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(testDataset,

        batch_size=args.test_batch_size, shuffle=False) #Shuffle should be false!
def logMagStft(numpyArray, sample_rate, n_fft):

    f, t, sx = sc.signal.stft(numpyArray, fs=sample_rate, nperseg=n_fft, noverlap=n_fft//2) 

    return np.log(np.abs(sx)+np.e**-10)



def computeMelspectrogram(numpyArray, sample_rate,  n_fft):

    S = librosa.feature.melspectrogram(y=numpyArray, sr=sample_rate, n_mels=256, fmax=8000, n_fft=n_fft, hop_length=n_fft//2)

    return np.log(S+1e-4)



sample_rate = 16000

number_of_examples_to_plot = 5

n_fft = 510

spectrograms = np.zeros((number_of_examples_to_plot, n_fft//2+1, int(2*64000/n_fft)+1))

for samples, instrumentsFamily in train_loader:

    for index in range(number_of_examples_to_plot):

        spectrograms[index] = computeMelspectrogram(samples[index].numpy(), sample_rate, n_fft)

    family = trainDataset.transformInstrumentsFamilyToString(instrumentsFamily.numpy().astype(int))

    break # SVM is only fitted to a fixed size of data



import matplotlib.pyplot as plt

    

for i in range(number_of_examples_to_plot):

    print(spectrograms[i].shape)

    plt.imshow(spectrograms[i])

    print(family[i])

    plt.colorbar()

    plt.show()
# CNN resnet architecture

class BasicBlock(nn.Module):

    expansion = 1



    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(self.expansion*planes)

            )



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = F.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):

        super(ResNet, self).__init__()

        self.in_planes = 64



        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(64)

        

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)

        layers = []

        for stride in strides:

            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)



    def forward(self, x):

        n_fft = 510

    

        spectrograms = np.zeros((len(x), n_fft//2+1, int(2*64000/n_fft)+1))

        for index, audio in enumerate(x.cpu().numpy()):

            spectrograms[index] = computeMelspectrogram(audio, 16000, n_fft)

        

        x = torch.from_numpy(spectrograms[:, np.newaxis, :, :]).to(device).float()



        

        out = F.relu(self.bn1(self.conv1(x)))

        out = F.max_pool2d(out, 2, 1)

        out = F.relu(self.bn2(self.conv2(out)))

        out = F.max_pool2d(out, 2, 2)



        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return F.log_softmax(out, dim=1)

    

def ResNet18():

    return ResNet(BasicBlock, [2,2,2,2])



def ResNet34():

    return ResNet(BasicBlock, [3,4,6,3])



def ResNet50():

    return ResNet(Bottleneck, [3,4,6,3])



def ResNet101():

    return ResNet(Bottleneck, [3,4,23,3])



def ResNet152():

    return ResNet(Bottleneck, [3,8,36,3])
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

            with torch.no_grad():

                print('F1 score: ' + str(sklearn.metrics.f1_score(target.cpu().numpy(), output.max(1)[1].cpu().numpy(), average='macro')))
# This function trains the model for one epoch

import sklearn

def valid(args, model, device, test_loader):

    model.eval()

    

    test_loss = 0

    f1_loss = []

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

            f1_loss.append(sklearn.metrics.f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='macro')) 

    test_loss /= len(test_loader.dataset)

    f1_loss = np.mean(f1_loss)



    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

    print("F1 valid loss: " + str(f1_loss))

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

torch.cuda.empty_cache()

model = ResNet18().to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, 

                      momentum=args.momentum)



for epoch in range(1, args.epochs + 1):

    train(args, model, device, train_loader, optimizer, epoch) 

    valid(args, model, device, valid_loader)

    test(args, model, device, test_loader, epoch)

ResNet18()