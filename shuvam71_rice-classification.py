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
print(os.listdir("../input/virtual-hack"))
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
data_dir = "../input/virtual-hack/car_data/car_data/train"

names = "../input/virtual-hack/names.csv"

annoTrain = "../input/virtual-hack/anno_train.csv"

anno = pd.read_csv(annoTrain)

anno.head()
# Transform the image (scaling, flipping and normalisation)

data_transforms = {

    'train': transforms.Compose([

        transforms.Resize((224, 224)),

       # transforms.RandomRotation(30),

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225])

    ]),

}





image_datasets = {x: datasets.ImageFolder(data_dir, data_transforms[x])

                  for x in ['train', 'valid']}



#info about no. of datapoints

image_datasets
valid_size = 0.2

batch_size = 64



length_train=len(image_datasets['train'])

indices=list(range(length_train))

split = int(np.floor(valid_size * length_train))



train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)

valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)



train_loader=torch.utils.data.DataLoader(image_datasets["train"],batch_size=batch_size,sampler=train_sampler,shuffle=False)

valid_loader=torch.utils.data.DataLoader(image_datasets["valid"],batch_size=batch_size,sampler=valid_sampler,shuffle=False)
pd_names = pd.read_csv(names)

print(pd_names.shape) # number of target output classes

pd_names
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

images, _ = next(iter(train_loader))

#print(train_loader.dataset.targets)

trainGrid = torchvision.utils.make_grid(images, nrow=8)



show(trainGrid)
# Make a grid from batch (for validation/test data)

images, _ = next(iter(valid_loader))

testGrid = torchvision.utils.make_grid(images, nrow=8)



show(testGrid)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
# Use GPU if it's available



model = models.resnet101(pretrained=True)

#print(model)



# Freeze parameters so we don't backprop through them

for param in model.parameters():

    param.requires_grad = False

    

num_ftrs = model.fc.in_features

model.fc = nn.Sequential(nn.Linear(num_ftrs, 196),

                         nn.LogSoftmax(dim=1))

    

# model.classifier = nn.Sequential(nn.Dropout(0.1),

#                                  nn.Linear(9216, 2048),

#                                  nn.ReLU(),

#                                  nn.Dropout(0.1),

#                                  nn.Linear(2048, 256),

#                                  nn.ReLU(),

#                                  nn.Linear(256, 196),

#                                  nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()



# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)



model.to(device)
def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):

  

    steps = 0

    running_loss = 0

    for e in range(epochs):

        

        # Model in training mode, dropout is on

        model.train()

        for images, labels in trainloader:

            

            steps += 1

            

            # Flatten images into a 784 long vector

            #images.resize_(images.size()[0], -1)

            

            images = images.to(device)

            labels = labels.to(device)

            

            #optimizer.zero_grad()

            

            output = model.forward(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            

            running_loss += loss.item()



            if steps % print_every == 0:

                # Model in inference mode, dropout is off

                model.eval()

                

                # Turn off gradients for validation, will speed up inference

                with torch.no_grad():

                    test_loss, accuracy = validation(model, testloader, criterion)

                

                print("Epoch: {}/{}.. ".format(e+1, epochs),

                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),

                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),

                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                

                running_loss = 0

                

                # Make sure dropout and grads are on for training

                model.train()



def validation(model, testloader, criterion):

  

    accuracy = 0

    test_loss = 0

    for images, labels in testloader:



        #images = images.resize_(images.size()[0], -1)

        

        images = images.to(device)

        labels = labels.to(device)



        output = model.forward(images)

        test_loss += criterion(output, labels).item()



        ## Calculating the accuracy 

        # Model's output is log-softmax, take exponential to get the probabilities

        ps = torch.exp(output)

        # Class with highest probability is our predicted class, compare with true label

        equality = (labels.data == ps.max(1)[1])

        # Accuracy is number of correct predictions divided by all predictions, just take the mean

        accuracy += equality.type_as(torch.FloatTensor()).mean()



    return test_loss, accuracy
# uncomment these two lines when you want to re-train



#train(model, train_loader, valid_loader, criterion, optimizer, epochs=30)

#validation(model, valid_loader, criterion)
# train_idx, valid_idx = indices[split:], indices[split:]

# train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)

# valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)



# train_loader=torch.utils.data.DataLoader(image_datasets["train"],batch_size=batch_size,sampler=train_sampler,shuffle=False)

# valid_loader=torch.utils.data.DataLoader(image_datasets["valid"],batch_size=batch_size,sampler=valid_sampler,shuffle=False)
# # Save the checkpoint



# checkpoint = {'input_size': num_ftrs,

#               'output_size': 196,

#               'epochs': 50,

#               'fc': model.fc,

#               'optimizer_state': optimizer.state_dict(),

#               'mapping': image_datasets['train'].class_to_idx,

#               'state_dict': model.state_dict()}



# # Save the checkpoint 

# torch.save(checkpoint, 'checkpoint.pth')

# # idx_to_class = {}

# # for x in image_datasets['train'].class_to_idx:

# #     idx_to_class[image_datasets['train'].class_to_idx[x]] = labelToName[x]
def load_checkpoint(filepath):

    print('Loading checkpoint...')

    checkpoint = torch.load(filepath, map_location='cpu')

    model = models.resnet101(pretrained=True)

    model.fc = checkpoint['fc']

    model.optimizer_state = checkpoint['optimizer_state']

    model.mapping = checkpoint['mapping']

    model.load_state_dict(checkpoint['state_dict'])

    print('Done')

    return model
model = load_checkpoint('checkpoint.pth')
!wget https://raw.githubusercontent.com/Iamsdt/60daysofudacity/master/day22/Helper.py
test_dir = "../input/virtual-hack/car_data/car_data/test"
test_transform = transforms.Compose([transforms.Resize(224),

                                     transforms.CenterCrop(224),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.485, 0.456, 0.406], 

                                                          [0.229, 0.224, 0.225])

                                    ]),