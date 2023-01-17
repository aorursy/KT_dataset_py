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

from torchvision import datasets, transforms,models

from torch.utils.data import DataLoader



data_dir = "../input/nonsegmentedv2"



mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



train_transform = transforms.Compose([

                                transforms.Resize(255),

                                transforms.RandomResizedCrop(224),

                                transforms.RandomHorizontalFlip(),

                                transforms.ColorJitter(),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



test_transform = transforms.Compose([

                                transforms.Resize(255),

                                transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



train_data = datasets.ImageFolder(data_dir, transform=train_transform)

test_data = datasets.ImageFolder(data_dir, transform=test_transform)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=200, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=200)



len(trainloader)
import os

data_dir = "../input/nonsegmentedv2"

classes = os.listdir(data_dir)

len(classes)
# visualize data

import numpy as np

import matplotlib.pyplot as plt



data_iter = iter(testloader)

images, labels = data_iter.next()



fig = plt.figure(figsize=(25, 5))

for idx in range(2):

    ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])

    # unnormolaize first

    img = images[idx] / 2 + 0.5

    npimg = img.numpy()

    img = np.transpose(npimg, (1, 2, 0)) #transpose

    ax.imshow(img, cmap='gray')

    title = classes[labels[idx]] + f"\tNumber: {idx}"

    ax.set_title(title)
images.shape
model = models.densenet161(pretrained=True)

model.classifier
# Freeze parameters

for param in model.parameters():

    param.requires_grad = False
import torch.nn as nn

from collections import OrderedDict



classifier = nn.Sequential(

  nn.Linear(in_features=2208, out_features=2208),

  nn.ReLU(),

  nn.Dropout(p=0.4),

  nn.Linear(in_features=2208, out_features=1024),

  nn.ReLU(),

  nn.Dropout(p=0.3),

  nn.Linear(in_features=1024, out_features=12),

  nn.LogSoftmax(dim=1)  

)

    

model.classifier = classifier

model.classifier
import torch.optim as optim

import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)



criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

# turn this off

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
import time



def train_model(model,

                train_loader,

                valid_loader,

                n_epochs,

                optimizer,

                scheduler,

                criterion,

                name="model.pt",

                path=None):

    # compare overfited

    train_loss_data, valid_loss_data = [], []

    # check for validation loss

    valid_loss_min = np.Inf

    # calculate time

    since = time.time()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    for epoch in range(n_epochs):

        print("Epoch: {}/{}".format(epoch + 1, n_epochs))

        # monitor training loss

        train_loss = 0.0

        valid_loss = 0.0

        total = 0

        correct = 0

        e_since = time.time()



        ###################

        # train the model #

        ###################

        model.train()  # prep model for training

        scheduler.step()  # step up scheduler

        for images, labels in train_loader:

            # Move input and label tensors to the default device

            images, labels = images.to(device), labels.to(device)

            # clear the gradients of all optimized variables

            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model

            log_ps = model(images)

            # calculate the loss

            loss = criterion(log_ps, labels)

            # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # perform a single optimization step (parameter update)

            optimizer.step()

            # update running training loss

            train_loss += loss.item() * images.size(0)



        ######################

        # validate the model #

        ######################

        print("\t\tGoing for validation")

        model.eval()  # prep model for evaluation

        for data, target in valid_loader:

            # Move input and label tensors to the default device

            data, target = data.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the loss

            loss_p = criterion(output, target)

            # update running validation loss

            valid_loss += loss_p.item() * data.size(0)

            # calculate accuracy

            proba = torch.exp(output)

            top_p, top_class = proba.topk(1, dim=1)

            equals = top_class == target.view(*top_class.shape)



            _, predicted = torch.max(output.data, 1)

            total += target.size(0)

            correct += (predicted == target).sum().item()



        # print training/validation statistics

        # calculate average loss over an epoch

        train_loss = train_loss / len(train_loader.dataset)

        valid_loss = valid_loss / len(valid_loader.dataset)



        # calculate train loss and running loss

        train_loss_data.append(train_loss * 100)

        valid_loss_data.append(valid_loss * 100)



        print("\tTrain loss:{:.6f}..".format(train_loss),

              "\tValid Loss:{:.6f}..".format(valid_loss),

              "\tAccuracy: {:.4f}".format(correct / total * 100))



        # save model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                valid_loss))

            torch.save(model.state_dict(), name)

            valid_loss_min = valid_loss

            # save to google drive

            if path is not None:

                torch.save(model.state_dict(), path)



        # Time take for one epoch

        time_elapsed = time.time() - e_since

        print('\tEpoch:{} completed in {:.0f}m {:.0f}s'.format(

            epoch + 1, time_elapsed // 60, time_elapsed % 60))



    # compare total time

    time_elapsed = time.time() - since

    print('Training completed in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))



    # return the model

    return [model, train_loss_data, valid_loss_data]
total_epoch = 20
model, train_loss, valid_loss = train_model(model, trainloader,

          testloader, total_epoch, optimizer,scheduler, criterion)
model.load_state_dict(torch.load('model.pt'))
plt.plot(train_loss, label="Training loss")

plt.plot(valid_loss, label="validation loss")

plt.legend(frameon=False)
def testModel(model, loader, device, criterion):

    

    test_loss = 0

    accuracy = 0



    with torch.no_grad():

        

        model.eval()



        for inputs, labels in loader:

            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)

            batch_loss = criterion(logps, labels)



            test_loss += batch_loss.item()



            # Calculate accuracy

            ps = torch.exp(logps)

            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



    print("Test Loss:{:.6f}".format(test_loss),

          "\nAccuracy: {:.4f}".format(accuracy / len(loader) * 100))
testModel(model, testloader, device, criterion)
def test2(model, loader, device, criterion):

    test_loss = 0.0

    class_correct = list(0. for i in range(102))

    class_total = list(0. for i in range(102))



    with torch.no_grad():

        model.eval()

        # iterate over test data

        for data, target in loader:

            # move tensors to GPU if CUDA is available

            data, target = data.to(device), target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the batch loss

            loss = criterion(output, target)

            # update test loss

            test_loss += loss.item() * data.size(0)

            # convert output probabilities to predicted class

            _, pred = torch.max(output, 1)

            # compare predictions to true label

            correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            # calculate test accuracy for each object class

            for i in range(len(target)):

                label = target.data[i]

                class_correct[label] += correct[i].item()

                class_total[label] += 1



    # average test loss

    test_loss = test_loss / len(loader.dataset)

    print('Test Loss: {:.6f}'.format(test_loss))



    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

        100. * np.sum(class_correct) / np.sum(class_total),

        np.sum(class_correct), np.sum(class_total)))
test2(model, testloader, device, criterion)
from PIL import Image



def test(file):

  ids = trainloader.dataset.class_to_idx



  with Image.open(file) as f:

      img = test_transform(f).unsqueeze(0)

      with torch.no_grad():

          out = model(img.to(device)).cpu().numpy()

          for key, value in ids.items():

              if value == np.argmax(out):

                    print(f"Predicted Label:Key {key} and value {value}")

          plt.imshow(np.array(f))

          plt.show()
classes[2]
file =  data_dir + "/Maize/106.png"

print(file)

print(f"Actual Label: {classes[2]}")

test(file)