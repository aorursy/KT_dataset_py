# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import time

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing all the packages



%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt

import torch

from torch import nn, optim

import torch.nn.functional as F

import torchvision

from torchvision import models, datasets, transforms

from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler
data_dir = "../input/car_data/car_data/"

# names = "../input/names.csv"

# annoTrain = "../input/anno_train.csv"

# anno = pd.read_csv(annoTrain)

# anno.head()
# Transform the image (scaling, flipping and normalisation)

data_transforms = {

    'train': transforms.Compose([

        transforms.Resize(255),

        transforms.RandomRotation(30),

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(20),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

    'test': transforms.Compose([

        transforms.Resize(255),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

}





# image_datasets = {x: datasets.ImageFolder(data_dir, data_transforms[x])

#                   for x in ['train', 'valid']}



# #info about no. of datapoints

# image_datasets
valid_size = 0.2

batch_size = 64



dataset = torchvision.datasets.ImageFolder(root=data_dir+"train", transform = data_transforms['train'])

trainloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True, num_workers = 2)



dataset2 = torchvision.datasets.ImageFolder(root=data_dir+"test", transform = data_transforms['test'])

testloader = torch.utils.data.DataLoader(dataset2, batch_size = 32, shuffle=False, num_workers = 2)

print(len(trainloader), len(testloader))
# pd_names = pd.read_csv(names)

# print(pd_names.shape) # number of target output classes

# pd_names
def show(image):

    if isinstance(image, torch.Tensor):

        image = image.numpy().transpose((1, 2, 0))

    else:

        image = np.array(image).transpose((1, 2, 0))

    # denormalisation

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    image = np.clip(image, 0, 1)

    # plot

    fig, Xaxis = plt.subplots(1, 1, figsize=(9, 9))

    %matplotlib inline

    plt.imshow(image)

    Xaxis.axis('off') 
# Make a grid from batch (for training data)

# This grid shows the images which are present in 1 batch

images, _ = next(iter(trainloader))

#print(train_loader.dataset.targets)

trainGrid = torchvision.utils.make_grid(images, nrow=8)



show(trainGrid)
# Make a grid from batch (for validation/test data)

images, _ = next(iter(testloader))

testGrid = torchvision.utils.make_grid(images, nrow=8)



show(testGrid)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
# Use GPU if it's available



model = models.resnet34(pretrained=True)



# Freeze parameters so we don't backprop through them

# for param in model.parameters():

#     param.requires_grad = False

    

model.fc = nn.Linear(512, 196)



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

model.to(device)
def train(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=5):

  

    losses = []

    accuracies = []

    test_accuracies = []

    # set the model to train mode initially

    model.train()

    for epoch in range(epochs):

        since = time.time()

        running_loss = 0.0

        running_correct = 0.0

        for i, data in enumerate(trainloader, 0):



            # get the inputs and assign them to cuda

            inputs, labels = data

            #inputs = inputs.to(device).half() # uncomment for half precision model

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            

            # calculate the loss/acc later

            running_loss += loss.item()

            running_correct += (labels==predicted).sum().item()



        epoch_duration = time.time()-since

        epoch_loss = running_loss/len(trainloader)

        epoch_acc = 100/32*running_correct/len(trainloader)

        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))

        

        losses.append(epoch_loss)

        accuracies.append(epoch_acc)

        

        # switch the model to eval mode to evaluate on test data

        model.eval()

        

        test_acc = validation(model, testloader, criterion)

        test_accuracies.append(test_acc)

        

        # re-set the model to train mode after validating

        model.train()

        scheduler.step(test_acc)

        since = time.time()

#         print(scheduler.get_lr())

    print('Finished Training')

    return model, losses, accuracies, test_accuracies

                

                # Make sure dropout and grads are on for training

#             model.train()    



def validation(model, testloader, criterion):

  

    correct = 0.0

    total = 0.0

    with torch.no_grad():

        for i, data in enumerate(testloader, 0):

            images, labels = data

            #images = images.to(device).half() # uncomment for half precision model

            images = images.to(device)

            labels = labels.to(device)

            

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



    test_acc = 100.0 * correct / total

    print('Accuracy of the network on the test images: %d %%' % (

        test_acc))

    return test_acc
submit = pd.read_csv('../input/sampleSubmission.csv')

submit.head()
# uncomment these two lines when you want to re-train



train(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=20)

# validation(model, valid_loader, criterion)
model.eval()

predict = []

for batch_i, (data, target) in enumerate(testloader):

    data, target = data.to(device), target.to(device)

    output = model(data)

    print(data)

    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        predict.append(i)



submit['Predicted'] = predict

submit.to_csv('submission.csv', index=False)

submit.head()

    

            
len(predict)
submit