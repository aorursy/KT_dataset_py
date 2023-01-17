import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision

from torchvision import datasets

from torchvision.transforms import transforms

from PIL import Image

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from torch.utils.data.sampler import SubsetRandomSampler



import math

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from IPython.display import HTML

import pandas as pd
train = pd.read_csv("../input/digit-recognizer/train.csv")
transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize((0.5,), (0.5,))

    ])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_digits(df):

    labels = []

    start_inx = 0

    if 'label' in df.columns:

        labels = [l for l in df.label.values]

        start_inx = 1

        

    

    digits = []

    for i in range(df.pixel0.size):

        digit = df.iloc[i].astype(float).values[start_inx:]

        digit = np.reshape(digit, (28,28))

        digit = transform(digit).type('torch.FloatTensor')

        if len(labels) > 0:

            digits.append([digit, labels[i]])

        else:

            digits.append(digit)



    return digits



trainX = get_digits(train)



  

batchSize  = 250 

validSize  = 0.2  





numTrain = len(trainX)

indices   = list(range(numTrain))

np.random.shuffle(indices)

split     = int(np.floor(validSize * numTrain))

trainIndex, validIndex = indices[split:], indices[:split]





from torch.utils.data.sampler import SubsetRandomSampler

trainSampler = SubsetRandomSampler(trainIndex)

validSampler = SubsetRandomSampler(validIndex)



trainLoad = torch.utils.data.DataLoader(trainX, batch_size=batchSize,

                    sampler=trainSampler)

validLoad = torch.utils.data.DataLoader(trainX, batch_size=batchSize, 

                    sampler=validSampler)



dataiter = iter(trainLoad)

images, labels = dataiter.next()

print(type(images))

print(images.shape)

print(labels.shape)
class myModel(nn.Module):

    

    def __init__(self):

        super(myModel, self).__init__()

        

        # The two Convolutional-Pooling Sets of Layers, 

        # connected through RELU functions.

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 30, kernel_size = 5, 

                      padding = 3),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 3, 

                         stride = 3)

            )

        

        self.layer2 = nn.Sequential(

            nn.Conv2d(30, 60, kernel_size = 5, 

                      padding = 3),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 3, 

                         stride = 3)

            )

        

        # Dropout 

        self.dropoutLayer = nn.Dropout()

        

        # Linear Layers

        self.linear1 = nn.Linear(4 * 4 * 60, 1100)

        self.linear2 = nn.Linear(1100, 500)

        self.linear3 = nn.Linear(500, 10)

    

    def forward(self, x):

        

        # Feed-forward network function

        net = self.layer1(x)

        net = self.layer2(net)

        net = net.reshape(net.size(0), -1) 

        net = self.dropoutLayer(net)

        net = self.linear1(net)

        net = self.linear2(net)

        

        return net
CNN = myModel()

lossFunc = nn.CrossEntropyLoss()

optimizer = optim.Adam(CNN.parameters(), lr = 0.0001)

totalLosses = []



for epoch in range(1, 21):

    # Training for twenty epochs

    

    lossAtEpoch = 0.0

    

    for index, data in enumerate(trainLoad, start = 0):

        inputs, names = data

        optimizer.zero_grad() # Zero the parameter gradients

        

        outputs = CNN(inputs)

        loss = lossFunc(outputs, names)

        

        loss.backward() # Make loss into Tensor

        optimizer.step()

        

        lossAtEpoch += loss.item()

        

        batchSize2 = 134 # Miniature Batch Size

        

        if index % batchSize2 == (batchSize2 - 1):

            print("Loss (Epoch " + str(epoch) + ") : " + str(lossAtEpoch / batchSize2))

            totalLosses.append(lossAtEpoch / batchSize2)

            

            lossAtEpoch = 0.0
arrayOfLosses = np.array(totalLosses)

arrayOfLosses.shape # =(38,0)

arrayOfLosses = arrayOfLosses.reshape(2,10)
plt.plot(totalLosses)

plt.title("Loss Function of CNN with MNIST Dataset")

plt.ylabel("Loss Function (CrossEntropyLoss)")

plt.xlabel("x-th Mini Batch (N = 250)")

plt.xticks(np.arange(len(totalLosses)), np.arange(1, len(totalLosses)+1))

plt.show()
df = pd.DataFrame({"CNN Loss (Optim = Adam)": totalLosses})

HTML(df.to_html(index = False, classes = "dataframe"))
numCorrect = [0. for i in range(10)]

numTotal = [0. for i in range(10)]



with torch.no_grad():

    for data in trainLoad:

        

        images, labels = data

        outputs = CNN(images)

        

        _, predicted = torch.max(outputs, 1)

        

        c = (predicted == labels).squeeze()

        

        for i in range(4):

            label = labels[i]

            numCorrect[label] += c[i].item()

            numTotal[label] += 1



# Average accuracy

avg = 0



for i in range(10):

    avg += 100 * numCorrect[i] / numTotal[i]

    print('Accuracy of %5s : %2d %%' % (

        i + 1, 100 * numCorrect[i] / numTotal[i]))



print("Average Accuracy: " + str(avg / 10))
# Define the test data loader

test        = pd.read_csv("../input/digit-recognizer/test.csv")

testX      = get_digits(test)

testLoad = torch.utils.data.DataLoader(testX, batch_size=batchSize)
ImageId = []

Label = []



# Loop through the data and get the predictions

for data in testLoad:

    # Move tensors to GPU if CUDA is available

    data = data.to(device)

    # Make the predictions

    output = CNN(data)

    # Get the most likely predicted digit

    _, pred = torch.max(output, 1)

    

    for i in range(len(pred)):        

        ImageId.append(len(ImageId)+1)

        Label.append(pred[i].cpu().numpy())



sub = pd.DataFrame(data={'ImageId':ImageId, 'Label':Label})

sub.describe

sub.to_csv("submission.csv", index = False)