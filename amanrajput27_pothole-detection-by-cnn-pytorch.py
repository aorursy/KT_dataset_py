import torch

if torch.cuda.is_available():

    device = torch.device("cuda:0")

else:

    device = "cpu"
import os

import pandas as pd

import cv2

import numpy as np

from tqdm import tqdm

from random import shuffle

import torch



normal_images  = []

potholes_images = []

path_normal = '/kaggle/input/pothole-detection-dataset/normal/'

path_potholes = '/kaggle/input/pothole-detection-dataset/potholes/'



for dirname, _, filenames in os.walk(path_normal):

    for filename in tqdm(filenames):

        try:

            img = cv2.imread(os.path.join(path_normal,filename) , cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img , (50,50))

            normal_images.append(np.array(img))

        except Exception as e:

            pass

        

for dirname, _, filenames in os.walk(path_potholes):

    for filename in tqdm(filenames):

        try:

            img = cv2.imread(os.path.join(path_potholes,filename) , cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img , (50,50))

            potholes_images.append(np.array(img))

        except Exception as e:

            pass

print(len(normal_images))

print(len(potholes_images))



processed_data = []

for img in normal_images:

    t = torch.LongTensor(1)

    t[0] = 0

    img = torch.FloatTensor(img)

    processed_data.append([img/255,t])

for img in potholes_images:

    t = torch.LongTensor(1)

    t[0] = 1

    img = torch.FloatTensor(img)

    processed_data.append([img/255,t])

                           

print(len(processed_data))

shuffle(processed_data)



train_data = processed_data[70:]

test_data = processed_data[0:70]



print(f"size of training data {len(train_data)}")

print(f"size of testing data {len(test_data)}")





import torch

import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1,32,5)

        self.conv2 = nn.Conv2d(32,64,5)

        self.conv3 = nn.Conv2d(64,128,5)

        

        x = torch.rand(1,50,50).view(-1,1,50,50)

        self.linear_in = None

        self.convs(x)

        

        self.fc1 = nn.Linear(self.linear_in,512)

        self.fc2 = nn.Linear(512,2)

        

    def convs(self,x):

        x = F.max_pool2d(F.relu(self.conv1(x)) , (2,2))

        x = F.max_pool2d(F.relu(self.conv2(x)) , (2,2))

        x = F.max_pool2d(F.relu(self.conv3(x)) , (2,2))

        

        if self.linear_in == None:

            self.linear_in = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        else:

            return x

    

    def forward(self,x):

        x = self.convs(x)

        x = x.view(-1,self.linear_in)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x , dim = 1)



net = Net()

net.to(device)

import torch.optim as optim



def train_model(Net , train_data):

    optimizer  = optim.Adam(net.parameters(),lr = 0.001)

    loss_function = nn.CrossEntropyLoss()



    for epoch in tqdm(range(10)):

        for i in (range(0,610,10)):

            batch = train_data[i:i+10]

            batch_x = torch.FloatTensor(10,1,50,50)

            batch_y = torch.LongTensor(10,1)



            for i in range(10):

                batch_x[i] = batch[i][0]

                batch_y[i] = batch[i][1]

            batch_x.to(device)

            batch_y.to(device)

            net.zero_grad()

            outputs = net(batch_x.view(-1,1,50,50))

            batch_y = batch_y.view(10)

            loss = F.nll_loss(outputs , batch_y)

            loss.backward()

            optimizer.step()

        print(f"epoch : {epoch}  loss : {loss}")
def test_model(Net,test_data):

    correct = 0

    total = 0



    with torch.no_grad():

        for data in tqdm(test_data):

            x = torch.FloatTensor(data[0])

            y = torch.LongTensor(data[1])



            x = x.view(-1,1,50,50)

            x = x.to(device)

            output = net(x)

            output = output.view(2)

            if(max(output[0],output[1]) == output[0]):

                index = 0

            else:

                index = 1

            if index == y[0]:

                correct += 1

            total += 1

        return round(correct/total , 5)

train_model(net,train_data)

acc = test_model(net,test_data)

print(acc)
