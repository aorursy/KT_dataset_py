# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

import torchvision

import torchvision.transforms as transforms



transform_image = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_set = torchvision.datasets.ImageFolder(root='../input/fruits-360_dataset/fruits-360/Training',

                                          transform=transform_image)

test_set = torchvision.datasets.ImageFolder(root='../input/fruits-360_dataset/fruits-360/Test', 

                                        transform=transform_image)

test_loader = torch.utils.data.DataLoader(test_set, 

                                         batch_size=1000,

                                         shuffle=True, 

                                         num_workers=2)

train_size1 = int(0.2 * len(train_set))

train_size2 = len(train_set) - train_size1

validation_set, new_train_set = torch.utils.data.random_split(train_set, [train_size1, train_size2])



train_loader = torch.utils.data.DataLoader(new_train_set, 

                                          batch_size=100,

                                          shuffle=True, 

                                          num_workers=2)



validation_loader = torch.utils.data.DataLoader(validation_set, 

                                          batch_size=100,

                                          shuffle=True, 

                                          num_workers=2)

num_epochs = 5 

num_classes = 103 

learning_rate = 0.001
import torch.nn as nn

import torch.nn.functional as F



class ConvNet(nn.Module): 

     def __init__(self): 

         super(ConvNet, self).__init__() 

         self.layer1 = nn.Sequential( nn.Conv2d(3, 100, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 

         self.layer2 = nn.Sequential( nn.Conv2d(100, 200, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)) 

         self.drop_out = nn.Dropout() 

         self.fc1 = nn.Linear(25 * 25 * 200, 500) 

         self.fc2 = nn.Linear(500, 103)

            

     def forward(self, x):

         out = self.layer1(x) 

         out = self.layer2(out) 

         out = out.reshape(out.size(0), -1) 

         out = self.drop_out(out) 

         out = self.fc1(out) 

         out = self.fc2(out) 

         return out

        

     def train(self):

         total_step = len(train_loader)

         loss_list = []

         acc_list = []

         for epoch in range(num_epochs):

            for i, (images, labels) in enumerate(train_loader):

                # Прямой запуск

                images = images.to(device)

                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                loss_list.append(loss.item())



                # Обратное распространение и оптимизатор

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



                # Отслеживание точности

                total = labels.size(0)

                _, predicted = torch.max(outputs.data, 1)

                correct = (predicted == labels).sum().item()

                acc_list.append(correct / total)



                if (i + 1) % 100 == 0:

                    with torch.no_grad():

                        correct = 0

                        total = 0

                        for images2, labels2 in validation_loader:

                            images2 = images.to(device)

                            labels2 = labels.to(device)

                            outputs2 = model(images2)

                            _, predicted = torch.max(outputs2.data, 1)

                            total += labels2.size(0)

                            correct += (predicted == labels2).sum().item()



                        print('Train Accuracy of the model: {} %'.format((correct / total) * 100))

                        print('Epochs =', epoch + 1)

                        if correct / total * 100 > 97:

                            return 
model = ConvNet()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)



model.train()
with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



    print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
import matplotlib.pyplot as plt

import numpy as np



def imshow(img):

    img = img / 2 + 0.5     # денормализуем

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()

    

imshow(torchvision.utils.make_grid(torch.Tensor.cpu(images[::12])))

print('Predicted classes', torch.Tensor.cpu(predicted[::12]))

print('Right classes', torch.Tensor.cpu(labels[::12]))