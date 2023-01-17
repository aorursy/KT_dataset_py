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
train_data_dir = '../input/fruits-360_dataset/fruits-360/Training/'

test_data_dir = '../input/fruits-360_dataset/fruits-360/Test/'



import os

len(list(os.walk(train_data_dir)))
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models
train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_data_dir, transform=train_transforms)

test_data = datasets.ImageFolder(test_data_dir, transform=test_transforms)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
model = models.densenet121(pretrained=True)
for param in model.parameters():

    param.requires_grad = False



from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([

                          ('fc1', nn.Linear(1024, 500)),

                          ('relu', nn.ReLU()),

                          ('fc2', nn.Linear(500, 2)),

                          ('output', nn.LogSoftmax(dim=1))

                          ]))

    

model.classifier = classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()



# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)



model.to(device);
epochs = 0

steps = 0

running_loss = 0

print_every = 5

for epoch in range(epochs):

    for inputs, labels in trainloader:

        steps += 1

        # Move input and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in testloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)

                    batch_loss = criterion(logps, labels)

                    

                    test_loss += batch_loss.item()

                    

                    # Calculate accuracy

                    ps = torch.exp(logps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    

            print(f"Epoch {epoch+1}/{epochs}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(testloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(testloader):.3f}")

            running_loss = 0

            model.train()
checkpoint = {'input_size': 784,

              'output_size': 10,

              'hidden_layers': [each.out_features for each in model.hidden_layers],

              'state_dict': model.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')