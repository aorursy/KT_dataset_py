import torch

import torch.nn.functional as F

from torch import nn, optim

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms, models

import matplotlib.pyplot as plt





import pandas as pd

import numpy as np



import os

print(os.listdir("../input"))
import torch

from torch.utils.data import Dataset

import pandas as pd

import numpy as np



class Dataset(Dataset):

    def __init__(self, path, transform=None):

        self.data = pd.read_csv(path)

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
from torchvision import transforms, models

from torch.utils.data import DataLoader



path = '../input/train.csv'



VALID_SIZE = 0.2



train_transform = transforms.Compose([

    transforms.ToPILImage(),

   # transforms.RandomRotation(0, 0.5),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



valid_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



train_data = Dataset(path, train_transform)

valid_data = Dataset(path, valid_transform)



trainloader = DataLoader(train_data, batch_size = 1500, shuffle = True)

testloader = DataLoader(valid_data, batch_size = 1500, shuffle = False)



len(trainloader)
import torch.nn.functional as F



def milan(input, beta=-0.25):

    '''

    Applies the Mila function element-wise:

    Mila(x) = x * tanh(softplus(1 + β)) = x * tanh(ln(1 + exp(x+β)))

    See additional documentation for mila class.

    '''

    return input * torch.tanh(F.softplus(input+beta))
import torch.nn as nn

from collections import OrderedDict



class mila(nn.Module):

    '''

    Applies the Mila function element-wise:

    Mila(x) = x * tanh(softplus(1 + β)) = x * tanh(ln(1 + exp(x+β)))

    Shape:

        - Input: (N, *) where * means, any number of additional

          dimensions

        - Output: (N, *), same shape as the input

    Examples:

        >>> m = mila(beta=1.0)

        >>> input = torch.randn(2)

        >>> output = m(input)

    '''

    def __init__(self, beta=-0.25):

        '''

        Init method.

        '''

        super().__init__()

        self.beta = beta



    def forward(self, input):

        '''

        Forward pass of the function.

        '''

        return milan(input, self.beta)
import torch.nn as nn



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),

            mila(),

            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, stride=2, padding=1),

            mila(),

            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(32, 64, 3, padding=1),

            mila(),

            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),

            mila(),

            nn.BatchNorm2d(64),

            #nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.conv3 = nn.Sequential(

            nn.Conv2d(64, 128, 3, padding=1),

            mila(),

            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),

            mila(),

            nn.BatchNorm2d(128),

            #nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.conv4 = nn.Sequential(

            nn.Conv2d(128, 256, 3, padding=1),

            mila(),

            nn.BatchNorm2d(256),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)

        )

        

        self.fc = nn.Sequential(

            nn.Linear(256, 256),

            nn.Dropout(0.3),

            mila(),

            nn.Linear(256, 256),

            nn.Dropout(0.4),

            mila(),

            nn.Linear(256, 10),

            nn.Softmax(dim=1)

        )

                

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        # flaten tensor

        x = x.view(x.size(0), -1)

        return self.fc(x)
!wget https://raw.githubusercontent.com/LiyuanLucasLiu/RAdam/master/cifar_imagenet/utils/radam.py
import radam

model = Net()



# Gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#move tensor to default device

model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = radam.RAdam(model.parameters(), lr=0.00159)
total_epoch = 50
n_epochs = total_epoch



#n_epochs = total_epoch



# compare overfited

train_loss_data,valid_loss_data = [],[]



# initialize tracker for minimum validation loss

valid_loss_min = np.Inf # set initial "min" to infinity



class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



for epoch in range(n_epochs):

    # monitor training loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train() # prep model for training

    for data, target in trainloader:

        # Move input and label tensors to the default device

        data, target = data.to(device), target.to(device)

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item() #*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval() # prep model for evaluation

    for data, target in testloader:

        # Move input and label tensors to the default device

        data, target = data.to(device), target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # update running validation loss 

        valid_loss += loss.item() #*data.size(0)

        # convert output probabilities to predicted class

        _, pred = torch.max(output, 1)

        # compare predictions to true label

        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # calculate test accuracy for each object class

        for i in range(16):

          label = target.data[i]

          class_correct[label] += correct[i].item()

          class_total[label] += 1

        

        

    # print training/validation statistics 

    # calculate average loss over an epoch

    train_loss = train_loss/len(trainloader.dataset)

    valid_loss = valid_loss/len(testloader.dataset)

    

    #clculate train loss and running loss

    train_loss_data.append(train_loss)

    valid_loss_data.append(valid_loss)

    

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch+1, 

        train_loss,

        valid_loss

        ))

    print('\t\tTest Accuracy: %4d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('\t\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'model.pt')

        valid_loss_min = valid_loss
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



plt.plot(train_loss_data, label='Training Loss')

plt.plot(valid_loss_data, label='Validation Loss')

plt.legend(frameon=False)
model.load_state_dict(torch.load('model.pt'))
# specify the image classes

classes = ['0', '1', '2', '3', '4',

           '5', '6', '7', '8', '9']



test_loss = 0.0

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



with torch.no_grad():

  model.eval()

  # iterate over test data

  for data, target in testloader:

      # move tensors to GPU if CUDA is available

      data, target = data.to(device), target.to(device)

      # forward pass: compute predicted outputs by passing inputs to the model

      output = model(data)

      # calculate the batch loss

      loss = criterion(output, target)

      # update test loss 

      test_loss += loss.item()*data.size(0)

      # convert output probabilities to predicted class

      _, pred = torch.max(output, 1)    

      # compare predictions to true label

      correct = np.squeeze(pred.eq(target.data.view_as(pred)))

      # calculate test accuracy for each object class

      for i in range(16):

          label = target.data[i]

          class_correct[label] += correct[i].item()

          class_total[label] += 1



# average test loss

test_loss = test_loss/len(testloader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(10):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
class DatasetSubmissionMNIST(torch.utils.data.Dataset):

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



submissionset = DatasetSubmissionMNIST('../input/test.csv', transform=transform)

submissionloader = torch.utils.data.DataLoader(submissionset, batch_size=128, shuffle=False)
submission = [['ImageId', 'Label']]



with torch.no_grad():

    model.eval()

    image_id = 1



    for images in submissionloader:

        images = images.to(device)

        log_ps = model(images)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1

            

print(len(submission) - 1)
import csv



with open('submission.csv', 'w') as submissionFile:

    writer = csv.writer(submissionFile)

    writer.writerows(submission)

    

print('Submission Complete!')