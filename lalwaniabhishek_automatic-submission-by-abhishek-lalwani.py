from __future__ import print_function, division

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

import os

import shutil

import pandas as pd

from torch.optim import lr_scheduler

import torchvision

import time

import os

import copy

import cv2

from sklearn.metrics import roc_auc_score

print("Imports done")
#Creating a training and a validation set folder

os.makedirs("../val")

os.makedirs("../train")

print("Validation set directory created")

print("Training set directory created")
#Creating the training and the validation set directories

full_dir = '../input/virtual-hack/car_data/car_data/train/'

train_dir = '../train/'

valid_dir = '../val/'

for dir in os.listdir(full_dir):

    os.makedirs(train_dir+dir)

    os.makedirs(valid_dir+dir)

print("Sub-Directories created")
#Deciding the split between validation and test data

split_factor = 0.9
#Initializing some directory variables

full_dir = '../input/virtual-hack/car_data/car_data/train/'

valid_dir = '../val/'

train_dir = '../train/'
#Making the appropriate directories for training and validation

for dir in os.listdir(full_dir):

    list = os.listdir(full_dir+dir) # dir is your directory path

    number_files = len(list)

    train_size = int(split_factor*number_files)

    valid_size = number_files - train_size

    for i in range(number_files):

        if (i < train_size):

            shutil.copy(full_dir+dir+'/'+list[i],train_dir+dir)

        else:

            shutil.copy(full_dir+dir+'/'+list[i],valid_dir+dir)

print("Data transfer done")
#Initializing the training and testing data loaders

data_transforms = {

    'train': transforms.Compose([

        transforms.Resize((224,224)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomAffine(360,translate = (0.125,0.125)),

        transforms.ColorJitter(0.2),

        transforms.ToTensor(),

        transforms.Normalize([0.388, 0.394, 0.401], [0.293, 0.285, 0.292])

    ]),

    'val': transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize([0.388, 0.394, 0.401], [0.293, 0.285, 0.292])

    ]),

}



data_dir = '../'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

                                          data_transforms[x])

                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,

                                             shuffle=True, num_workers=4)

              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#defining imshow for visualizing the data

def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.388, 0.394, 0.401])

    std = np.array([0.293, 0.285, 0.292])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated





# Get a batch of training data

inputs, classes = next(iter(dataloaders['train']))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs)



imshow(out, title=[class_names[x] for x in classes])
#Defining the classifier and the relevant parameters

# Use GPU if it's available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.resnet101(pretrained=True)



# Freeze parameters so we don't backprop through them

for param in model.parameters():

    param.requires_grad = False

    

model.fc = nn.Sequential(nn.Linear(2048, 512),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(512, 196),

                                 nn.LogSoftmax(dim=1))



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



model.to(device);
#defining the method which is used for training the model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                scheduler.step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
#Helper method for visualizing the peformance of the model

def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()



    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)

            labels = labels.to(device)



            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)



            for j in range(inputs.size()[0]):

                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)

                ax.axis('off')

                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                imshow(inputs.cpu().data[j])



                if images_so_far == num_images:

                    model.train(mode=was_training)

                    return

        model.train(mode=was_training)
#Training the model

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,

                       num_epochs=40)
#Visualizing the performance of the model

visualize_model(model_ft)
#Generating the list of images

test_dir = '../input/virtual-hack/car_data/car_data/test/'

folder_list = []

for folder in os.listdir(test_dir):

    folder_list.append(folder)

folder_list.sort()

name_list = []

true_label_list = []

for i in range(len(folder_list)):

    current_list = []

    current_label_list = []

    for j in os.listdir(test_dir+folder_list[i]):

        current_list.append(j)

        current_label_list.append(i)

    current_list.sort()

    name_list.extend(current_list)

    true_label_list.extend(current_label_list)

print('Name list generated')

print('True label list generated')
#Generating the test predictions

test_transforms = transforms.Compose([transforms.Resize((224,224)),

                                transforms.ToTensor(),

                                transforms.Normalize([0.388, 0.394, 0.401], [0.293, 0.285, 0.292])])

predict = []

values = []

model.eval()

test_data = datasets.ImageFolder('../input/virtual-hack/car_data/car_data/test', transform = test_transforms)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

with torch.no_grad():

        for i, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)

            vals, preds = torch.max(outputs, 1)

            predict.extend(preds.tolist())

            values.extend(vals.tolist())

print('Predictions generated')

print('Values generated')
#Printing id and predictions

final_result = []

for i in range (len(predict)):

    filename = str(name_list[i])[0:5]

    pred_category  = predict[i]

    final_result.append((filename, pred_category))

final_result
#Saving the prediction in a csv

final_output = pd.DataFrame(final_result, columns=["Id", "Predicted"])

final_output.to_csv('final_output.csv', index=False)
#Checking whether the CSV is generated properly or not

test_csv = pd.read_csv("final_output.csv")

test_csv
#Getting the auc score

from sklearn.preprocessing import LabelBinarizer

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    lb = LabelBinarizer()

    lb.fit(y_test)

    y_test = lb.transform(y_test)

    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)

multiclass_roc_auc_score(true_label_list, predict)