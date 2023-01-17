#import all the necessary packages

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data import DataLoader,Subset

from torchvision import models,transforms,datasets



import os

print(os.listdir("../input"))
#checking GPU avaliablity

gpu_yes = torch.cuda.is_available()



if gpu_yes:

    print('GPU is ready.')

else:

    print('No GPU found. Using CPU.')
#loading data 

batch_size  = 32



data_dir = '../input/cell_images/cell_images'



#define data transformation 

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),

                                      transforms.RandomRotation(30),

                                      transforms.RandomHorizontalFlip(),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])



test_transform = transforms.Compose([transforms.Resize((224,224)),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.485, 0.456, 0.406],

                                                          [0.229, 0.224, 0.225])])





trainset = datasets.ImageFolder(data_dir, transform = train_transform)

validset = datasets.ImageFolder(data_dir, transform = test_transform)

testset = datasets.ImageFolder(data_dir, transform = test_transform)



#randomly spliting the data into training set, validation set, and test set

num_train = len(trainset)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor((0.7 * num_train)))

valid_split = int(np.floor((num_train-split)*0.5))



train_idx = indices[:split]

valid_idx = indices[split:(split+valid_split)]

test_idx = indices[(split+valid_split):]



#loading the data based on the split index

trainset = Subset(trainset, train_idx)

validset = Subset(validset, valid_idx)

testset = Subset(testset,test_idx)



trainloader = DataLoader(trainset,  batch_size = batch_size, num_workers=0)

validloader = DataLoader(validset,  batch_size = batch_size, num_workers=0)

testloader = DataLoader(testset,  batch_size = batch_size,drop_last=True, num_workers=0)

#loading the pretrained model 

model = models.vgg16(pretrained = True)



#Freeze the parameters for the model

for param in model.parameters():

    param.requires_grad = False



#view all the layer on VGG16 

print(model)
#define the classifier 

class Classifier(nn.Module):

    

    def __init__(self):

        super(Classifier, self).__init__()

        

        self.hidden1 = nn.Linear(25088,4096)

        self.hidden2 = nn.Linear(4096, 4096)

        self.output = nn.Linear(4096, 2)

        

        self.dropout = nn.Dropout(0.5)

        

    def forward(self, x):

        

        x = self.dropout(F.relu(self.hidden1(x)))

        x = self.dropout(F.relu(self.hidden2(x)))

        x = self.output(x)

        

        return x
#replace the model's default 

model.classifier = Classifier()



print(model)



#

if gpu_yes:

    model.cuda()



#define the lost function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.classifier.parameters(), lr = 0.001, momentum = 0.9)
#training the model



epoches = 10



valid_loss_min = np.Inf



torch.cuda.manual_seed_all(2019)



for epoch in range(1,epoches+1):

    

    train_loss = 0.0

    valid_loss = 0.0

    

    #training pharse

    model.train()

    for data, target in trainloader:

        

        if gpu_yes:

            data, target = data.cuda(), target.cuda()

            

        optimizer.zero_grad()

        output = model(data)

        

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()



        train_loss += loss.item()*data.size(0)

    

    #validation pharse

    model.eval()

    with torch.no_grad():

        for data, target in validloader:

            

            if gpu_yes:

                data, target = data.cuda(), target.cuda()

                

            output = model(data)

            loss = criterion(output, target)



            valid_loss += loss.item()*data.size(0)

            

    train_loss = train_loss/len(trainloader.dataset)

    valid_loss = valid_loss/len(validloader.dataset)

    

    #print out the loss and save the model if the validation loss decreases

    print('Epoch: {}\tTraining  Loss : {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

    

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}. Saving model...)'.format(valid_loss_min, valid_loss))

        torch.save(model.state_dict(), 'model.pt')

        valid_loss_min = valid_loss

        

        
#loading the optimized model from the the training session

model.load_state_dict(torch.load('model.pt'))
#define the categories the dataset contains

cat_to_name = ['Parasitized','Uninfected']
#testing the model



test_loss = 0.0

class_correct = list(0. for i in range(2))

class_total = list(0. for i in range(2))



model.eval()

with torch.no_grad():

    for data, target in testloader:

        if gpu_yes:

            data, target = data.cuda(), target.cuda()



        output = model(data)

        #print(target.data[1])



        loss = criterion(output,target)

        test_loss += loss.item()*data.size(0)



        _, pred = torch.max(output, 1)

        correct_tensor = pred.eq(target.data.view_as(pred))

        correct = np.squeeze(correct_tensor.numpy()) if not gpu_yes else np.squeeze(

                             correct_tensor.cpu().numpy())



        for i in range(batch_size):

            label = target.data[i]

            class_correct[label] += correct[i].item()

            class_total[label] += 1



    

test_loss = test_loss/len(testloader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(2):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %(

              cat_to_name[i], 100 * class_correct[i] / class_total[i],

              np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Acccuracy of %5s: N/A (no training example)' %

             (cat_to_name[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

       100 *np.sum(class_correct) / np.sum(class_total),

       np.sum(class_correct), np.sum(class_total)))