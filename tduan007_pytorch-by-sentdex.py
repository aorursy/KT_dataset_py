import torch



x = torch.Tensor([5,3])

y = torch.Tensor([2,1])



print(x*y)
x = torch.zeros([2,5])

print(x)
print(x.shape)
y = torch.rand([2,5])

print(y)
y.view([1,10])
y = y.view([1,10])

y
import torch

import torchvision

from torchvision import transforms, datasets



train = datasets.MNIST('', train=True, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))



test = datasets.MNIST('', train=False, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)
# 10 NUMBER

# data var will be saved

for data in trainset:

    print(data)

    break
# pic and first pic; number and first number

X, y = data[0][0], data[1][0]
print(data[1])
X.shape
y.shape
import matplotlib.pyplot as plt  # pip install matplotlib



plt.imshow(data[0][0].view(28,28))

plt.show()
data[0][0][0][0]
data[0][0][0][3]
total = 0

counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}





for data in trainset:

    Xs, ys = data

    for y in ys:

        counter_dict[int(y)] += 1

        total += 1



print(counter_dict)



for i in counter_dict:

    print(f"{i}: {counter_dict[i]/total*100.0}%")
import torch

import torchvision

from torchvision import transforms, datasets



train = datasets.MNIST('', train=True, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))



test = datasets.MNIST('', train=False, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))





trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)
import torch.nn as nn

import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)

        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 10)



    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return F.log_softmax(x, dim=1)



net = Net()

print(net)
X = torch.randn((28,28))

#X = X.view(1,28*28)

X = X.view(-1,28*28)
output = net(X)

output
import torch

import torchvision

from torchvision import transforms, datasets

import torch.nn as nn

import torch.nn.functional as F



train = datasets.MNIST('', train=True, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))



test = datasets.MNIST('', train=False, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor()

                       ]))





trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)





class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)

        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 10)



    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return F.log_softmax(x, dim=1)



net = Net()

print(net)
import torch.optim as optim



loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(3): # 3 full passes over the data

    for data in trainset:  # `data` is a batch of data

        X, y = data  # X is the batch of features, y is the batch of targets.

        net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.

        output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)

        loss = F.nll_loss(output, y)  # calc and grab the loss value

        #loss = criterion(outputs, y)

        loss.backward()  # apply this loss backwards thru the network's parameters

        optimizer.step()  # attempt to optimize weights to account for loss/gradients

    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
correct = 0

total = 0



with torch.no_grad():

    for data in testset:

        X, y = data

        output = net(X.view(-1,784))

        #print(output)

        for idx, i in enumerate(output):

            #print(torch.argmax(i), y[idx])

            if torch.argmax(i) == y[idx]:

                correct += 1

            total += 1



print("Accuracy: ", round(correct/total, 3))
import matplotlib.pyplot as plt



plt.imshow(X[3].view(28,28))

plt.show()
print(torch.argmax(net(X[3].view(-1,784))[0]))
import matplotlib.pyplot as plt



plt.imshow(X[4].view(28,28))

plt.show()
print(torch.argmax(net(X[4].view(-1,784))[0]))
import urllib.request

import os

import zipfile

# small data

urllib.request.urlretrieve("https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip","cats_and_dogs_filtered.zip")

# big data

#urllib.request.urlretrieve("https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip","cats_and_dogs_filtered.zip")
!ls
local_zip = 'cats_and_dogs_filtered.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('../output/')
import os

os.listdir('../output')
import os

os.listdir('../output/cats_and_dogs_filtered')
import os

os.listdir("../output/cats_and_dogs_filtered/train")
import os

os.listdir("../output/cats_and_dogs_filtered/validation")
import os

l=len(os.listdir("../output/cats_and_dogs_filtered/validation/cats"))

l
import os

import cv2

import numpy as np

from tqdm import tqdm
class DogsVSCats():

    IMG_SIZE = 50

    CATS = "../output/cats_and_dogs_filtered/train/cats"

    DOGS = "../output/cats_and_dogs_filtered/train/dogs"

    TESTING = "../output/cats_and_dogs_filtered/validation"

    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
REBUILD_DATA = True # set to true to one once, then back to false unless you want to change something in your training data.



class DogsVSCats():

    IMG_SIZE = 50

    CATS = "../output/cats_and_dogs_filtered/train/cats"

    DOGS = "../output/cats_and_dogs_filtered/train/dogs"

    TESTING = "../output/cats_and_dogs_filtered/validation"

    LABELS = {CATS: 0, DOGS: 1}

    training_data = []



    catcount = 0

    dogcount = 0



    def make_training_data(self):

        for label in self.LABELS:

            print(label)

            for f in tqdm(os.listdir(label)):

                if "jpg" in f:

                    try:

                        path = os.path.join(label, f)

                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 

                        #print(np.eye(2)[self.LABELS[label]])



                        if label == self.CATS:

                            self.catcount += 1

                        elif label == self.DOGS:

                            self.dogcount += 1



                    except Exception as e:

                        pass

                        #print(label, f, str(e))



        np.random.shuffle(self.training_data)

        np.save("training_data.npy", self.training_data)

        print('Cats:',dogsvcats.catcount)

        print('Dogs:',dogsvcats.dogcount)



if REBUILD_DATA:

    dogsvcats = DogsVSCats()

    dogsvcats.make_training_data()
IMG_SIZE = 50

CATS = "../output/cats_and_dogs_filtered/train/cats"

DOGS = "../output/cats_and_dogs_filtered/train/dogs"

TESTING = "../output/cats_and_dogs_filtered/validation"

LABELS = {CATS: 0, DOGS: 1}

training_data = []



catcount = 0

dogcount = 0
for label in LABELS:

    print(label)

    for f in tqdm(os.listdir(label)):

        if "jpg" in f:

            try:

                path = os.path.join(label, f)

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                training_data.append([np.array(img), np.eye(2)[LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 

                #print(np.eye(2)[self.LABELS[label]])



                if label == CATS:

                    catcount += 1

                elif label == self.DOGS:

                    dogcount += 1

    

            except Exception as e:

                pass

                #print(label, f, str(e))

    np.random.shuffle(training_data)

    np.save("training_data.npy", training_data)

    print('Cats:',dogsvcats.catcount)

    print('Dogs:',dogsvcats.dogcount)
training_data = np.load("training_data.npy", allow_pickle=True)

print(len(training_data))
import torch



X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)

X = X/255.0

y = torch.Tensor([i[1] for i in training_data])
import matplotlib.pyplot as plt



plt.imshow(X[0], cmap="gray")
# cat

print(y[0])
training_data = np.load("training_data.npy", allow_pickle=True)

print(len(training_data))
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)

X = X/255.0

y = torch.Tensor([i[1] for i in training_data])
VAL_PCT = 0.1  # lets reserve 10% of our data for validation

val_size = int(len(X)*VAL_PCT)

print(val_size)
train_X = X[:-val_size]

train_y = y[:-val_size]



test_X = X[-val_size:]

test_y = y[-val_size:]



print(len(train_X), len(test_X))
# model 

import torch

import torch.nn as nn

import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self):

        super().__init__() # just run the init of parent class (nn.Module)

        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window

        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window

        self.conv3 = nn.Conv2d(64, 128, 5)



        x = torch.randn(50,50).view(-1,1,50,50)

        self._to_linear = None

        self.convs(x)



        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.

        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).



    def convs(self, x):

        # max pooling over 2x2

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))



        if self._to_linear is None:

            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x



    def forward(self, x):

        x = self.convs(x)

        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 

        x = F.relu(self.fc1(x))

        x = self.fc2(x) # bc this is our output layer. No activation here.

        return F.softmax(x, dim=1)





net = Net()

print(net)
import torch.optim as optim



optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_function = nn.MSELoss()
# train data

BATCH_SIZE = 100

EPOCHS = 10



for epoch in range(EPOCHS):



    for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev

        #print(f"{i}:{i+BATCH_SIZE}")

        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)

        batch_y = train_y[i:i+BATCH_SIZE]



        net.zero_grad()



        outputs = net(batch_X)

        loss = loss_function(outputs, batch_y)

        loss.backward()

        optimizer.step()    # Does the update

        



    

    ####################################################

    print(f"Epoch: {epoch}. Loss: {loss}")

  
# test data



correct = 0

total = 0

with torch.no_grad():

    for i in tqdm(range(len(test_X))):

        real_class = torch.argmax(test_y[i])

        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list, 

        predicted_class = torch.argmax(net_out)



        if predicted_class == real_class:

            correct += 1

        total += 1

print("Accuracy: ", round(correct/total, 3))