import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import os

import time

import csv
train_on_gpu = torch.cuda.is_available()
class MNIST(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        item = self.data.iloc[index]

                

        image = item[1:].values.astype(np.uint8).reshape((28, 28))

        label = item[0]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
BATCH_SIZE = 128

VALID_SIZE = 0.15



transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])

dataset = pd.read_csv('../input/digit-recognizer/train.csv')

train = MNIST(dataset, transform=transform)

valid = MNIST(dataset, transform=transform)



num_train = len(train)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(VALID_SIZE * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, sampler=valid_sampler)



print("Train length-")

print(len(train_idx))

print("Valid length-")

print(len(valid_idx))
class Net(nn.Module):

    def __init__(self): 

        super(Net, self).__init__()

        self.cnn_model = nn.Sequential(

            nn.Conv2d(1, 10, 5),         # (N, 1, 28, 28) -> (N,  10, 24, 24)

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),  # (N, 10, 24, 24) -> (N,  10, 12, 12)

            nn.Conv2d(10, 20, 5),        # (N, 10, 12, 12) -> (N, 20, 8, 8)  

            nn.Dropout2d(),

            nn.ReLU(),

            nn.MaxPool2d(2, stride=2)   # (N,20, 8, 8) -> (N, 20, 4, 4)

        )

        self.fc_model = nn.Sequential(

            nn.Linear(320,50),         # (N, 320) -> (N, 50)

            nn.ReLU(),

            nn.Linear(50,10)            # (N, 50)  -> (N, 10)

        )

        

    def forward(self, x):

        x = self.cnn_model(x)

        x = x.view(x.size(0), -1)

        x = self.fc_model(x)

        return x
model = Net()

if train_on_gpu:

    model.cuda()
LEARNING_RATE = 0.01



criterion = nn.CrossEntropyLoss()

momentum = 0.5

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,

                                momentum=momentum)
%%time

epochs = 150

valid_loss_min = np.Inf

train_losses, valid_losses = [], []



for epoch in range(1, epochs+1):

    running_loss = 0



    for images, labels in train_loader:

        if train_on_gpu:

            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    else:

        valid_loss = 0

        accuracy = 0

        with torch.no_grad():

            model.eval()

            for images, labels in valid_loader:

                if train_on_gpu:

                    images, labels = images.cuda(), labels.cuda()

                output = model(images)

                _, top_class = output.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                

                valid_loss += criterion(output, labels)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

        model.train()

        

        train_losses.append(running_loss/len(train_loader))

        valid_losses.append(valid_loss/len(valid_loader))

        

        improved = valid_loss < valid_loss_min



        if epoch == 1 or epoch % 5 == 0 or improved:

            print(f"Epoch: {epoch}/{epochs}.. ",

                  f"Training Loss: {running_loss/len(train_loader):.3f}.. ",

                  f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. ",

                  f"Valid Accuracy: {accuracy/len(valid_loader):.3f}")

        

        if improved:

            valid_loss_min = valid_loss

            torch.save(model.state_dict(), 'mnist.pt')

            print('saving current model')
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)
model.load_state_dict(torch.load('mnist.pt'))
class SubmissionMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        image = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28, 1))



        

        if self.transform is not None:

            image = self.transform(image)

            

        return image
transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



testset = SubmissionMNIST('../input/digit-recognizer/test.csv', transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

submission = [['ImageId', 'Label']]



with torch.no_grad():

    model.eval()

    image_id = 1



    for images in testloader:

        if train_on_gpu:

            images = images.cuda()

        output = model(images)

        ps = torch.exp(output)

        top_p, top_class = output.topk(1, dim=1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1
with open('prediction.csv', 'w') as File:

    writer = csv.writer(File)

    writer.writerows(submission)