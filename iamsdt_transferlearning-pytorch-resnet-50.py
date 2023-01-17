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

from torchvision import datasets, transforms, models

from torch.autograd import Variable

import numpy as np

from matplotlib import pyplot as plt
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



data_dir = "../input/flowers_/flowers_/"



train_transform = transforms.Compose([

                                transforms.Resize(150),

                                transforms.RandomResizedCrop(150),

                                transforms.RandomHorizontalFlip(),

                                transforms.ColorJitter(),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



test_transform = transforms.Compose([

                                transforms.Resize(150),

                                transforms.CenterCrop(150),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



train_data  = datasets.ImageFolder(data_dir + '/train', train_transform)

test_data = datasets.ImageFolder(data_dir + '/test', test_transform)
classes = train_data.classes

class_idx = train_data.class_to_idx

class_idx
train_loader = torch.utils.data.DataLoader(train_data,

                                           batch_size=100,

                                           shuffle=True,

                                           num_workers=1)



test_loader = torch.utils.data.DataLoader(test_data,

                                          batch_size=100,

                                          shuffle=False,

                                          num_workers=1)



print(len(train_loader))
images , labels = next(iter(train_loader))

images.shape, len(labels)
# visualize data

import numpy as np

import matplotlib.pyplot as plt



data_iter = iter(test_loader)

images, labels = data_iter.next()



fig = plt.figure(figsize=(25, 5))

for idx in range(2):

    ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])

    # unnormolaize first

    img = images[idx] / 2 + 0.5

    npimg = img.numpy()

    img = np.transpose(npimg, (1, 2, 0)) #transpose

    ax.imshow(img, cmap='gray')

    ax.set_title(classes[labels[idx]])
model = models.resnet50(pretrained=True)

model.fc
for param in model.parameters():

    param.required_grad = False
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2048, out_features=1024),

  nn.LeakyReLU(),

  nn.Dropout(p=0.2),

  nn.Linear(in_features=1024, out_features=512),

  nn.LeakyReLU(),

  nn.Dropout(p=0.3),

  nn.Linear(in_features=512, out_features=5),

  nn.LogSoftmax(dim=1)  

)

    

model.fc = classifier

model.fc
import torch.optim as optim

import torch



criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

#gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
n_epochs = 5



# compare overfited

train_loss_data,valid_loss_data = [],[]



# initialize tracker for minimum validation loss

valid_loss_min = np.Inf # set initial "min" to infinity



class_correct = list(0. for i in range(5))

class_total = list(0. for i in range(5))



for epoch in range(n_epochs):

    # monitor training loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train() # prep model for training

    for data, target in train_loader:

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

    for data, target in test_loader:

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

    train_loss = train_loss/len(train_loader.dataset)

    valid_loss = valid_loss/len(test_loader.dataset)

    

    #clculate train loss and running loss

    train_loss_data.append(train_loss)

    valid_loss_data.append(valid_loss)

    

    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch+1,

        n_epochs,

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
# load the saved model

model.load_state_dict(torch.load('model83.pt'))
# save model

torch.save(model.state_dict(), 'model83.pt')
# check for overfitting

plt.plot(train_loss_data, label = "taining loss")

plt.plot(valid_loss_data, label = "validation loss")

plt.legend(frameon = False)
# track test loss



total_class = 5



test_loss = 0.0

class_correct = list(0. for i in range(total_class))

class_total = list(0. for i in range(total_class))



with torch.no_grad():

  model.eval()

  # iterate over test data

  for data, target in test_loader:

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

test_loss = test_loss/len(test_loader.dataset)

print('Test Loss: {:.6f}\n'.format(test_loss))



for i in range(total_class):

    if class_total[i] > 0:

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
import os

from glob import glob

import numpy as np  # linear algebra

import torch

from PIL import Image

from matplotlib import pyplot as plt



def showPred(path):

    classes = train_loader.dataset.class_to_idx



    file = glob(os.path.join(path, '*.jpg'))[0]



    with Image.open(file) as f:

        img = test_transform(f).unsqueeze(0)

        with torch.no_grad():

            out = model(img.to(device)).cpu().numpy()

            for key, value in classes.items():

                if value == np.argmax(out):

                    print(key)

        plt.imshow(np.array(f))

        plt.show()
rose = data_dir + "test/rose/"

showPred(rose)
sun = data_dir + "test/sunflower/"

showPred(sun)