import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

from torch import nn, optim

from torchvision import transforms, datasets, models

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim import lr_scheduler

from collections import OrderedDict

import seaborn as sns

from PIL import Image



import os

print(os.listdir("../input/mias-jpeg/MIAS-JPEG/"))
# Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(size=(224,224,), scale=(0.08, 1.0), 

                                                                   ratio=(0.75, 1.3333333333333333), 

                                                                   interpolation=2),

                                       transforms.RandomVerticalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406], 

                                                            [0.229, 0.224, 0.225])])

                                



test_transforms = transforms.Compose([transforms.RandomGrayscale(p=0.1),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406], 

                                                           [0.229, 0.224, 0.225])])



validation_transforms = transforms.Compose([transforms.RandomGrayscale(p=0.1),

                                            transforms.CenterCrop(224),

                                            transforms.ToTensor(),

                                            transforms.Normalize([0.485, 0.456, 0.406], 

                                                                 [0.229, 0.224, 0.225])])

img_dir='../input/mias-jpeg/MIAS-JPEG/'

train_data = datasets.ImageFolder(img_dir,transform=train_transforms)
# number of subprocesses to use for data loading

num_workers = 0

# percentage of training set to use as validation

valid_size = 0.15



test_size = 0.1



# convert data to a normalized torch.FloatTensor

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])



# obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

valid_split = int(np.floor((valid_size) * num_train))

test_split = int(np.floor((valid_size+test_size) * num_train))

valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]



print(len(valid_idx), len(test_idx), len(train_idx))



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

test_sampler = SubsetRandomSampler(test_idx)



# prepare data loaders (combine dataset and sampler)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,

    sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=32, 

    sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(train_data, batch_size=20, 

    sampler=test_sampler, num_workers=num_workers)
plt.bar(['Total_Data','Train_Data','Valid_Data','Test_Data'],[len(train_data),len(train_idx),len(valid_idx),len(test_idx)])
model =models.resnet50(pretrained=True)



for param in model.parameters():

    param.requires_grad = False



model.fc = nn.Linear(2048, 2, bias=True)



fc_parameters = model.fc.parameters()



for param in fc_parameters:

    param.requires_grad = True

    

model
use_cuda = torch.cuda.is_available()

if use_cuda:

    model = model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.fc.parameters(), lr=0.001 , momentum=0.9)
def train(n_epochs, model, optimizer, criterion, use_cuda, save_path):

    """returns trained model"""

    # initialize tracker for minimum validation loss

    valid_loss_min = np.Inf

    train_correct = 0.

    train_total = 0.

    train_accuracy=[]

    

    valid_correct=0.

    valid_total=0.

    valid_accuracy=[]

    

    loss_valid=[]

    loss_train=[]

    for epoch in range(1, n_epochs+1):

        # initialize variables to monitor training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

        

        ###################

        # train the model #

        ###################

        model.train(True)

        for batch_idx, (data, target) in enumerate(train_loader):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()



            # initialize weights to zero

            optimizer.zero_grad()

            

            output = model(data)

            

            # calculate loss

            loss = criterion(output, target)

            

            # back prop

            loss.backward()

            

            # grad

            optimizer.step()

            

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            

            

            pred = output.data.max(1, keepdim=True)[1]

            # compare predictions to true label

            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            train_total += data.size(0)

            train_accuracy.append(100. * train_correct / train_total) 

            loss_train.append(train_loss)

        

            if batch_idx % 100 == 0:

                print('Epoch %d, Batch %d loss: %.6f' %

                  (epoch, batch_idx + 1, train_loss))

        

        ######################    

        # validate the model #

        ######################

        model.train(False)

        model.eval()

        for batch_idx, (data, target) in enumerate(valid_loader):

            # move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            ## update the average validation loss

            output = model(data)

            loss = criterion(output, target)

            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            

            

            pred = output.data.max(1, keepdim=True)[1]

            # compare predictions to true label

            valid_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

            valid_total += data.size(0)

            valid_accuracy.append(100. * valid_correct / valid_total) 

            

            loss_valid.append(valid_loss)   

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

            epoch, 

            train_loss,

            valid_loss

            ))

        

        ## TODO: save the model if validation loss has decreased

        if valid_loss < valid_loss_min:

            torch.save(model.state_dict(), save_path)

            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

            valid_loss_min,

            valid_loss))

            valid_loss_min = valid_loss

    plt.plot(train_accuracy)

    plt.plot(valid_accuracy)

    plt.title('Train Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Batch_idx')

    plt.legend(['train', 'validation'])

    plt.show()         

    

    plt.plot(loss_train)

    plt.plot(loss_valid)

    plt.title('Train Loss')

    plt.ylabel('Loss')

    plt.xlabel('Batch_idx')

    plt.legend(['train', 'validation'])

    plt.show()         

    

    print('\Train Accuracy: %4d%% (%2d/%2d)' % (

        100. * train_correct / train_total, train_correct, train_total))

    # return trained model

    return model
train(15, model, optimizer, criterion, use_cuda, 'tumour_detection.pt')
model.load_state_dict(torch.load('tumour_detection.pt'))
def test(model, criterion, use_cuda):



    # monitor test loss and accuracy

    test_loss = 0.

    correct = 0.

    total = 0.

    accuracy=[]

    loss_test=[]

    for batch_idx, (data, target) in enumerate(test_loader):

        # move to GPU

        if use_cuda:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # update average test loss 

        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        loss_test.append(test_loss)

        # convert output probabilities to predicted class

        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label

        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

        total += data.size(0)

        accuracy.append(100. * correct / total)    

    plt.plot(loss_test)

    plt.title('Test Loss')

    plt.ylabel('Loss')

    plt.xlabel('Batch_idx')

    #plt.legend(['train', 'validation'])

    plt.show()         

    

    plt.plot(accuracy)

    plt.title(' Accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Batch_idx')

    #plt.legend(['train', 'validation'])

    plt.show()    

    

    print('Test Loss: {:.6f}\n'.format(test_loss))



    print('\nTest Accuracy: %4d%% (%2d/%2d)' % (

        100. * correct / total, correct, total))

test(model, criterion, use_cuda)
!!jupyter nbconvert *.ipynb