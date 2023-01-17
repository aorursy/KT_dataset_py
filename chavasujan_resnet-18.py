import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch as tch

import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models



import os

print(os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images"))



device = tch.device("cuda:0" if tch.cuda.is_available() else "cpu")

print("Current  Device:",device)
image_folder = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images"



#Define Classes for the title

classes= ['Parasitized', 'Uninfected']
#Import the images data with just Tensor Transformation for viewing the images

image_data = datasets.ImageFolder(root=image_folder,transform = transforms.ToTensor())



#Create a place holder for images and labels to study

image_list= []

title_list = []



#Create a random list if Prasitized nad Uninfected (limiting samples to 10)

np.random.seed(2019)

for i in np.random.choice(range(1,len(image_data)),10):

    image_list.append(image_data[i][0].numpy().transpose())

    title_list.append(classes[image_data[i][1]])



fig = plt.figure(figsize=(25, 15))

for i,(image,title) in enumerate(zip(image_list,title_list)):

    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[], title=title)

    plt.imshow(image)

plt.show()    

#Import the images data with just Tensor Transformation for viewing the images

transformation  = transforms.Compose([transforms.Resize((120,120)),

                                     transforms.ColorJitter(0.1),

                                     transforms.RandomHorizontalFlip(),

                                     transforms.RandomVerticalFlip(),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])





image_data = datasets.ImageFolder(root=image_folder,transform = transformation)



#Create a place holder for images and labels to study

image_list= []

title_list = []



#Create a random list if Prasitized nad Uninfected (limiting samples to 10)

np.random.seed(2019)

for i in np.random.choice(range(1,len(image_data)),10):

    image_list.append(image_data[i][0].numpy().transpose())

    title_list.append(classes[image_data[i][1]])



fig = plt.figure(figsize=(25, 15))

for i,(image,title) in enumerate(zip(image_list,title_list)):

    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[], title=title)

    plt.imshow(image)

plt.show()    

len(image_data)
import torch.nn as nn

import torch.nn.functional as F



model = models.resnet18(pretrained=True)
model.fc 
model.fc = nn.Linear(512,2)
for params in pretr_model.parameters():

    params.requires_grad = False



for params in pretr_model.parameters():

    params.requires_grad = True
model.to(device)

loss_function = nn.CrossEntropyLoss()

learning_rate = 0.001

#Using SGD Optimization

optimizer = tch.optim.SGD(model.parameters(),lr= learning_rate)
print(model)
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader



test_size= 0.2

samples = len(image_data)

indices = list(range(samples))

np.random.shuffle(indices)

train_len  =  int(np.floor(samples * (1-test_size)))

train_index, test_index = indices[:train_len], indices[train_len:]



train_sampler,test_sampler = SubsetRandomSampler(train_index),SubsetRandomSampler(test_index)



train_loader = DataLoader(image_data,sampler= train_sampler,batch_size = 64)

test_loader = DataLoader(image_data, sampler= test_sampler,batch_size = 64, )
num_epochs = 10

for epoch in range(num_epochs):

    train_loss= 0.0

    #Explicitly start model training

    model.train()

    

    for i, (images,labels) in enumerate(train_loader):

        images,labels = images.to(device),labels.to(device)

        input_data = images.view(-1,3,120,120)

        output_data  = model(input_data)

        

        optimizer.zero_grad()

        loss = loss_function(output_data, labels)

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item() * 64 #batch_size for train

        

    print("Epoch: {} - Loss:{:.4f}".format(epoch+1,train_loss / len(train_loader.dataset)))
tch.save(model.state_dict(), "Resnet_SGD.pytorch")
sample_model = model

sample_model.load_state_dict(tch.load("Resnet_SGD.pytorch"))

sample_model.eval()
actual = []

predict = []



model.eval()    # explicitly stating the testing 

with tch.no_grad():

    for images, labels in test_loader:

        images, labels = images.to(device), labels.to (device)

        actual.extend(labels.data.tolist())



        test = images.view(-1, 3, 120, 120)

        outputs = model(test)

        predicted = tch.max(outputs, 1)[1]

        predict.extend(predicted.data.tolist())
from sklearn.metrics import confusion_matrix, precision_score, recall_score,accuracy_score, f1_score



print(confusion_matrix(actual,predict))

print("Accuracy =",accuracy_score(actual,predict))

print("Precision =",precision_score(actual,predict))

print("Recall =",recall_score(actual,predict))

print("F1 Score =",f1_score(actual,predict))