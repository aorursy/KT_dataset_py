import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import time

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data1 = pd.read_csv('../input/Kannada-MNIST/train.csv')

data2 = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
data = pd.concat([data1, data2], axis = 0)
target = data.pop('label')



data = data/255.0



data['label'] = target



del data1

del data2

del target
class CNNUnit(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(CNNUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,

                              out_channels=out_channels,

                             kernel_size = 3, stride = 1, padding = 1)

        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        self.LR = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        

    def forward(self,x):

        output = self.LR(self.bn(self.conv(x)))

        return output
class FCUnit(nn.Module):

    def __init__(self, input_dim, output_dim, initrange):

        super(FCUnit, self).__init__()

        self.fc = nn.Linear(in_features=input_dim,

                           out_features = output_dim)

        nn.init.uniform_(a = -initrange, b = initrange, tensor=self.fc.weight)

        nn.init.zeros_(self.fc.bias)

        self.LR = nn.LeakyReLU(inplace=True)

        self.bn = nn.BatchNorm1d(num_features=output_dim)

        

    def forward(self, x):

        output = self.bn(self.LR(self.fc(x)))

        return output
class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

       #image --> (_,1,28,28)

        

        self.unit1 = CNNUnit(in_channels=1, out_channels=32)

        self.unit2 = CNNUnit(in_channels=32, out_channels=32)

        self.unit3 = CNNUnit(in_channels=32, out_channels=32)

        

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride = 2)

        #image --> (_,32,14,14)

        

        self.unit4 = CNNUnit(in_channels=32, out_channels=64)

        self.unit5 = CNNUnit(in_channels=64, out_channels=64)

        self.unit6 = CNNUnit(in_channels=64, out_channels=64)

        self.unit7 = CNNUnit(in_channels=64, out_channels=64)

        

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride = 2)

        #image --> (_,64,7,7)

        

        self.unit8 = CNNUnit(in_channels=64, out_channels=128)

        self.unit9 = CNNUnit(in_channels=128, out_channels=128)

        

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride = 2, padding = 1)

        #image --> (_,128,4,4)

        

        self.unit10 = CNNUnit(in_channels=128, out_channels=256)

        

        self.avg = nn.AvgPool2d(kernel_size=2, stride = 2)

        #image --> (_,256,2,2)

        

        #flatten --> (2*2*256)

        

        self.fc1 = FCUnit(input_dim=2*2*256, output_dim=64, initrange=0.5)

        self.fc2 = nn.Linear(64,10)

        

        self.net = nn.Sequential(self.unit1,self.unit2,self.unit3,self.pool1,

                                self.unit4,self.unit5,self.unit6,self.unit7,self.pool2,

                                self.unit8,self.unit9,self.pool3,

                                self.unit10,self.avg)



        

        #Spatial_transformer

        self.localization = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=7),

            nn.MaxPool2d(2, stride=2),

            nn.ReLU(True),

            nn.Conv2d(8, 10, kernel_size=5),

            nn.MaxPool2d(2, stride=2),

            nn.ReLU(True))

        

        # Regressor for the 3 * 2 affine matrix

        self.fc_loc = nn.Sequential(

            nn.Linear(10 * 3 * 3, 32),

            nn.ReLU(True),

            nn.Linear(32, 3 * 2))



        # Initialize the weights/bias with identity transformation

        self.fc_loc[2].weight.data.zero_()

        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        

        

    # Spatial transformer network forward function

    def stn(self, x):

        xs = self.localization(x)

        xs = xs.view(-1, 10 * 3 * 3)

        theta = self.fc_loc(xs)

        theta = theta.view(-1, 2, 3)



        grid = F.affine_grid(theta, x.size())

        x = F.grid_sample(x, grid)



        return x

        

        

    def forward(self, image):

        #transform the unit

        output = self.stn(image)

        

        #perform usual forwarding

        output = self.net(output)

        output = output.view(output.size(0),-1)

        output = F.dropout((self.fc1(output)), p = 1-KEPT_PROB, training = False)

        output = F.log_softmax(self.fc2(output))

        

        return output
class oversampdata(Dataset):

    def __init__(self, data):

        self.data = torch.FloatTensor(data.values.astype('float'))

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        target = self.data[index][-1]

        data_val = self.data[index][:-1]

        return data_val, target
def train_func(sub_train_):



    # Train the model

    train_loss = []

    train_acc = []

    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True)

    for i, (image, label) in enumerate(data):

        optimizer.zero_grad()

        image, label = image.to(device), label.to(device)

        image = image.view(-1,1,28,28)

        label = label.long()

        output = model(image)

        loss = criterion(output, label)

        train_loss.append(float(loss.item()))

        loss.backward()

        optimizer.step()

        del(image)

        output = output.data.max(dim=1)[1]

        train_acc.append(((output.data == label.data).float().mean()).item())

        del(label)

        torch.cuda.empty_cache()

        if i % 200 == 0:

            print("Batch = {},\t loss = {:2.4f},\t accuracy = {}%".format(i, train_loss[-1], train_acc[-1]*100))



    # Adjust the learning rate

    #schedular.step()



def test(data_):

    loss = 0

    acc = 0

    data = DataLoader(data_, batch_size=BATCH_SIZE)

    for image, label in data:

        image, label = image.to(device), label.to(device)

        image = image.view(-1,1,28,28)

        label = label.long()

        with torch.no_grad():

            output = model(image)

            loss = criterion(output, label)

            loss += float(loss.item())

            del(image)

            output = output.data.max(dim=1)[1]

            acc += float((output.data == label.data).sum().item() / len(data_))

            del(label)

            torch.cuda.empty_cache()



    return loss / len(data_), acc
BATCH_SIZE = 32

KEPT_PROB = 0.5

N_EPOCHS = 10

LEARNING_RATE = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = CNN().to(device)



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
print(device)
train_data, test_data = train_test_split(data, test_size = 0.05, random_state = 128)
sub_train = oversampdata(train_data)

sub_test = oversampdata(test_data)
for epoch in range(N_EPOCHS):



    start_time = time.time()

    train_func(sub_train)

    valid_loss, valid_acc = test(sub_test)



    secs = int(time.time() - start_time)

    mins = secs / 60

    secs = secs % 60



    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))

    print('\tLoss: {valid_loss:.4f}(test)\t|\tAcc: {valid_acc:.1f}%(test)\t|'.

         format(valid_loss = valid_loss, valid_acc = valid_acc * 100))
test_data = pd.read_csv('../input/Kannada-MNIST/test.csv')

test_data = test_data.drop(['id'], axis = 1)

test_data = test_data/255.0

test_data = torch.FloatTensor(test_data.values.astype('float'))

test_data = test_data.view(-1,1,28,28)
def predict(data):

    model.eval()

    predictions = []

    for image in  data:

        image = image.view(1,1,28,28)

        image = image.to(device)

        output = model(image)

        del image

        predictions.append(int(output.argmax(1)))

        

    return predictions
predictions = predict(test_data)
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission = submission.drop(['label'], axis = 1)

submission['label'] = predictions

submission.to_csv('submission.csv', index=False)