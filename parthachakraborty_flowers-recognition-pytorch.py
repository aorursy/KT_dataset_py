!apt-get -qq install -y graphviz && pip install -q pydot

!pip install torchvision

!pip install torchviz

!pip install torchsummary
import numpy as np

from matplotlib import pyplot as plt

import torch

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision.datasets as dests

from torch.utils import data

from torch.autograd import Variable

from torchvision import models

from torchsummary import summary

from torchviz import make_dot

import pandas as pd

import os

from shutil import copyfile

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from PIL import Image

from numpy import asarray

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
INITIAL_DATA_DIRECTORY = "/kaggle/input/flowers-recognition/flowers/"

DATA_DIRECTORY = "flower"

labels = []

labelencoder = LabelEncoder()

labelencoder.fit(os.listdir(INITIAL_DATA_DIRECTORY))

if not os.path.exists(DATA_DIRECTORY): os.makedirs(DATA_DIRECTORY)

for item in os.listdir(INITIAL_DATA_DIRECTORY):

    if item != "flowers":

        for element in os.listdir(INITIAL_DATA_DIRECTORY + "/" + item):

          if ".py" not in element:

            if not os.path.exists(DATA_DIRECTORY[0:] +  "/" + element):

                copyfile(INITIAL_DATA_DIRECTORY[0:] + "/" + item + "/" + element, DATA_DIRECTORY[0:] +  "/" + element)

            labels.append((element, labelencoder.transform([item]).tolist()[0]))

df = pd.DataFrame(data=labels,columns=['id', 'label']).sample(frac=1).reset_index(drop=True)
class FlowerDataset(data.Dataset):

  'Characterizes a dataset for PyTorch'

  def __init__(self,root, list_IDs, labels, transform=None):

        'Initialization'

        self.root = root

        self.labels = labels

        self.list_IDs = list_IDs

        self.transform = transform



  def __len__(self):

        'Denotes the total number of samples'

        return len(self.list_IDs)



  def __getitem__(self, index):

        'Generates one sample of data'

        # Select sample

        ID = self.list_IDs[index]



        # Load data and get label

        X = Image.open(self.root + ID)



        if self.transform:

            for transform_item in self.transform:

                X = transform_item(X)

        y = self.labels[index]



        return X, y

image_height, image_width = 400, 400

train, test = train_test_split(df, test_size=0.2)

train_dataset = FlowerDataset(root="flower/", list_IDs=train['id'].tolist(), labels= train['label'].tolist(), transform=  [transforms.Resize(size=(image_height, image_width),interpolation=Image.BILINEAR),transforms.ToTensor()])

test_dataset = FlowerDataset(root="flower/",list_IDs=test['id'].tolist(), labels= test['label'].tolist(), transform=  [transforms.Resize(size=(image_height, image_width),interpolation=Image.BILINEAR),transforms.ToTensor()])





batch_size = 100

n_iters = 3000

num_epochs = int(n_iters / (len(train_dataset)/batch_size))





train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle=True)




class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        self.conv_layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),

                                nn.ReLU(),

                                nn.MaxPool2d(kernel_size=2),

                                nn.Dropout2d(p=0.1)

                                )

        self.conv_layer_2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),

                                     nn.ReLU(),

                                     nn.MaxPool2d(kernel_size=4),

                                     nn.Dropout2d(p=0.05)

                                     )

        self.conv_layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),

                                     nn.ReLU(),

                                     nn.MaxPool2d(kernel_size=5),

                                     nn.Dropout2d(p=0.05)

                                     )

        self.fully_connected_layer_1 = nn.Linear(64 * (int)((image_height/(2*4*5)) * (image_width/(2*4*5))), 1000) 

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

        self.fully_connected_layer_2 = nn.Linear(1000, len(labelencoder.classes_.tolist()))



    def forward(self, x):

        out = self.conv_layer_1(x)

        out = self.conv_layer_2(out)

        out = self.conv_layer_3(out)

        out = out.view(out.size(0), -1)

        out = self.fully_connected_layer_1(out)

        out = self.dropout(out)

        out = self.relu(out)

        out = self.fully_connected_layer_2(out)

        return out
input_dim = image_height*image_width

hidden_dim = 50

output_dim = 10

model = CNNModel()

torch.cuda.empty_cache() 

if torch.cuda.is_available():

    model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


images,labels = next(iter(train_loader))

if torch.cuda.is_available():

    images = Variable(images.cuda())

else:

    images = Variable(images)

output = model(images)

print(images[0].size())

summary(model, input_size=(3,image_height,image_width))

make_dot(output.mean(),params=dict(model.named_parameters()))
iter_counter = 0

torch.cuda.empty_cache() 

for epoch in range(num_epochs):

    for i, (images,labels) in enumerate(train_loader):

        if torch.cuda.is_available():

            images = Variable(images.cuda())

            labels = Variable(labels.cuda())

        else:

            images = Variable(images)

            labels = Variable(labels)



        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        iter_counter +=1

        #print("Iter {} completed".format(iter_counter))

        if iter_counter% 500 ==0:

            correct = 0

            total = 0

            for images, labels in test_loader:

                if torch.cuda.is_available():

                    images = Variable(images.cuda())

                else:

                    images = Variable(images)

                outputs = model(images)

                _, predicted = torch.max(outputs.data,1)

                total += labels.size(0)

                if torch.cuda.is_available():

                    correct += (predicted.cpu() == labels.cpu()).sum()

                else:

                    correct += (predicted == labels).sum()

                

            accuracy = (100 *correct)/total

            print("Iteration: {} Loss: {} Accuracy: {}".format(iter_counter, loss, accuracy))
torch.save(model,"model_weight.pth")
model = torch.load("model_weight.pth")

if torch.cuda.is_available():

    model.cuda()

predicted_label = []

true_label = []

for images, labels in test_loader:

        if torch.cuda.is_available():

            images = Variable(images.cuda())

        else:

            images = Variable(images)

        torch.cuda.empty_cache() 

        outputs = model(images)

        _, predicted = torch.max(outputs.data,1)

        predicted_label += predicted.cpu().numpy().tolist()

        true_label += labels.cpu().numpy().tolist()
inverse_true_label = labelencoder.inverse_transform(true_label)

inverse_predicted_label = labelencoder.inverse_transform(predicted_label)

multilabel_confusion_matrix(inverse_true_label, inverse_predicted_label, labels=labelencoder.classes_.tolist())