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

import torch.nn as nn

import torch.optim as optim

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import json

import cv2
train_data_dir = "../input/flower_data/flower_data/train/"

test_data_dir = "../input/flower_data/flower_data/valid/"

file_path = "../input/cat_to_name.json"
df = pd.read_json(file_path, typ='series')

df = df.to_frame('category')

df.head()
df.count()
img_dir = os.path.join(train_data_dir, str(np.random.randint(1, 103)))



for img_name in os.listdir(img_dir)[1:2]:

    img_array = cv2.imread(os.path.join(img_dir, img_name))

    img_array = cv2.resize(img_array, (225,225), interpolation=cv2.INTER_AREA)

    plt.imshow(img_array)

    plt.show()

    print(img_array.shape)
train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(225),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_data_dir, transform=train_transforms)

test_data = datasets.ImageFolder(test_data_dir, transform=test_transforms)
print("Number of Training images: ", len(train_data))

print("Number of Test images: ", len(test_data))
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
model1 = models.densenet121(pretrained=True)
for params in model1.features.parameters():

    params.requires_grad = False
model1.classifier = nn.Sequential(nn.Linear(1024, 256),

                                  nn.ReLU(),

                                  nn.Dropout(0.2),

                                  nn.Linear(256, 102),

                                  nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()



optimizer = optim.Adam(model1.classifier.parameters(), lr=0.003)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1.to(device)

device
epochs = 25

steps = 0

running_loss = 0

print_every = 50

training_loss = []

Test_loss = []

for e in range(epochs):

    for images, labels in trainloader:

        steps += 1

        

        images, labels = images.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        y_pred = model1.forward(images)

        loss = criterion(y_pred, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        accuracy = 0

        if steps % print_every == 0:

            test_loss = 0

            

            model1.eval()

            

            with torch.no_grad():

                for images, labels in testloader:

                    images, labels = images.to(device), labels.to(device)

                    y_pred = model1.forward(images)

                    batch_loss = criterion(y_pred, labels)

                    

                    test_loss += batch_loss.item()

                    

                    ps = torch.exp(y_pred)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                

                

                  

                print(f"Epoch {e+1}/{epochs}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(testloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(testloader):.3f}")

                

                training_loss.append(running_loss)

                running_loss = 0

                model1.train()

            Test_loss.append(test_loss)
torch.save(model1.state_dict(), 'model1.pth')
data_dir = "../input/test set/"



data = datasets.ImageFolder(data_dir, transform=test_transforms)

dataloader = torch.utils.data.DataLoader(data)
data_labels = []

model1.to(device)

model1.eval()

with torch.no_grad():

    for images, labels in dataloader:

        images = images.to(device)

        ps = model1.forward(images)

        

        if type(ps) == tuple:

            ps, _ = ps

        

        _, preds_tensor = torch.max(ps, 1)

        preds = np.squeeze(preds_tensor.numpy())if not device else np.squeeze(preds_tensor.cpu().numpy())

        data_labels.append(int(preds))
files =[]

categories = []

for file in os.listdir(os.path.join(data_dir, "test set")):

    files.append(file)



for i in data_labels:

    categories.append(df.loc[i+1, 'category'])    
d = {'Image_Name': files, 'Class_Label': data_labels, 'Flower_Category': categories}

result = pd.DataFrame(d)
result = result.sort_values(by="Image_Name")
result
result.to_csv("../working/result.csv")