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
!ls ../working/
from __future__ import print_function

from __future__ import division

import torch

import torch.nn as nn

import torch.optim as optim

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

import json

from sklearn.metrics import roc_curve

print("PyTorch Version: ",torch.__version__)

print("Torchvision Version: ",torchvision.__version__)
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):

    since = time.time()



    val_acc_history = []



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

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

                    # Get model outputs and calculate loss

                    # Special case for inception because in training it has an auxiliary output. In train

                    #   mode we calculate the loss by summing the final output and the auxiliary output

                    #   but in testing we only consider the final output.

                    if is_inception and phase == 'train':

                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

                        outputs, aux_outputs = model(inputs)

                        loss1 = criterion(outputs, labels)

                        loss2 = criterion(aux_outputs, labels)

                        loss = loss1 + 0.4*loss2

                    else:

                        outputs = model(inputs)

                        loss = criterion(outputs, labels)



                    _, preds = torch.max(outputs, 1)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'valid':

                val_acc_history.append(epoch_acc)



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts) 

    

    return model, val_acc_history, outputs, labels
def set_parameter_requires_grad(model, feature_extracting):

    if feature_extracting:

        for param in model.parameters():

            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    # Initialize these variables which will be set in this if statement. Each of these

    #   variables is model specific.

    model_ft = None

    input_size = 0



    if model_name == "resnet":

        """ Resnet18

        """

        model_ft = models.resnet18(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224



    elif model_name == "alexnet":

        """ Alexnet

        """

        model_ft = models.alexnet(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 224



    elif model_name == "vgg":

        """ VGG11_bn

        """

        model_ft = models.vgg11_bn(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 224



    elif model_name == "squeezenet":

        """ Squeezenet

        """

        model_ft = models.squeezenet1_0(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

        model_ft.num_classes = num_classes

        input_size = 224



    elif model_name == "densenet":

        """ Densenet

        """

        model_ft = models.densenet121(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier.in_features

        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        input_size = 224



    elif model_name == "inception":

        """ Inception v3

        Be careful, expects (299,299) sized images and has auxiliary output

        """

        model_ft = models.inception_v3(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        # Handle the auxilary net

        num_ftrs = model_ft.AuxLogits.fc.in_features

        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        # Handle the primary net

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs,num_classes)

        input_size = 299



    else:

        print("Invalid model name, exiting...")

        exit()



    return model_ft, input_size

# Top level data directory. Here we assume the format of the directory conforms

#   to the ImageFolder structure

data_dir = "../input/car_data/car_data/"

#test_data_dir = "../input/hackathon-blossom-flower-classification/"



# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]

model_name = "inception"



# Number of classes in the dataset

num_classes = 196



# Batch size for training (change depending on how much memory you have)

batch_size = 16



# Number of epochs to train for

num_epochs = 75



# Flag for feature extracting. When False, we finetune the whole model,

#   when True we only update the reshaped layer params

feature_extract = False
# Initialize the model for this run

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)



# Print the model we just instantiated

#print(model_ft)

print("Model Loading Process Done")
data_transforms = {

    'transform': transforms.Compose([

        transforms.RandomResizedCrop(input_size),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

}



#print(len(datasets.ImageFolder(data_dir + '/train')))

full_dataset = datasets.ImageFolder(data_dir + '/train', data_transforms['transform'])



train_size = int(0.8 * len(full_dataset))

test_size = len(full_dataset) - train_size

#print(train_size)

#print(test_size)

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])



image_datasets = {

    'train':train_dataset, 

    'valid':test_dataset

}

dataloaders_dict = {

    'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),

    'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=4)

}
# Detect if we have a GPU available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
# Send the model to GPU

model_ft = model_ft.to(device)



# Gather the parameters to be optimized/updated in this run. If we are

#  finetuning we will be updating all parameters. However, if we are

#  doing feature extract method, we will only update the parameters

#  that we have just initialized, i.e. the parameters with requires_grad

#  is True.

params_to_update = model_ft.parameters()

#print("Params to learn:")

if feature_extract:

    params_to_update = []

    for name,param in model_ft.named_parameters():

        if param.requires_grad == True:

            params_to_update.append(param)

            #print("\t",name)

            pass

else:

    for name,param in model_ft.named_parameters():

        if param.requires_grad == True:

            #print("\t",name)

            pass



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

#optimizer_ft = optim.Adam(params_to_update,lr=0.0001)
# Setup the loss fxn

criterion = nn.CrossEntropyLoss()



# Train and evaluate

model_ft, hist, out, lab = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=75, is_inception=(model_name=="inception"))
def save_checkpoint():

    checkpoint = {

        'model':model_ft, 

        'state_dict':model_ft.state_dict(),

        'optimizer':optimizer_ft.state_dict()

    }

    torch.save(checkpoint, '../working/checkpoint.pt')

def load_checkpoint(filepath, inference = False):

    checkpoint = torch.load(filepath + 'checkpoint.pt')

    model = checkpoint['model']

    if inference:

        for parameter in model.parameter():

            parameter.require_grad = False

        model.eval()

    model.to(device)

    return model
save_checkpoint()
model_ft = load_checkpoint(filepath='../working/')
test_data_dir ="../input/car_data/car_data/"

data_transforms = {

    'testing': transforms.Compose([

        transforms.Resize(input_size),

        transforms.CenterCrop(input_size),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

}

test_image_datasets = {x: datasets.ImageFolder(os.path.join(test_data_dir, 'test'), data_transforms[x]) for x in ['testing']}

test_dataloaders_dict = {x: torch.utils.data.DataLoader(test_image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['testing']}
import glob

files = glob.glob("../input/car_data/car_data/test/")

#len(files)
# path = '../input/car_data/car_data/test/Ram C/V Cargo Van Minivan 2012/*.*'

# files = glob.glob(path)

# print(len(files))
data = []

with open("../input/names.csv", 'r') as G:

     data.append(G.read())



data = data[0].split("\n")

data.remove('')

data = [item.replace("Ram C/V Cargo Van Minivan 2012", "Ram C-V Cargo Van Minivan 2012") for item in data]

#print(data)

#d = {index(i) for i in data: }

from os import listdir

from os.path import isfile, join



ids = []

op = []

#print(data)

for i in data:

    path = "../input/car_data/car_data/test/"+ str(i)+"/*.*"

    #mypath = "../input/car_data/car_data/test/"+ str(i)+"/"

    #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    files = glob.glob(path)

    for j in files:

        ids.append(j.split('/')[-1].split('.')[0])

        op.append(data.index(i) + 1)

    #print(len(files))

    #print(len(onlyfiles))

output = []

for inputs, labels in test_dataloaders_dict['testing']:

    inputs = inputs.to(device)

    labels = labels.to(device)

    outputs = model_ft(inputs)

    _, predicted = torch.max(outputs, 1)

    for i in predicted:

        output.append(int(i)+1)

#output
sample_submission = pd.read_csv('../input/sampleSubmission.csv')

#ids = list(range(1, len(output)+1))

submission = pd.DataFrame({

    'Id' :ids,

    'Predicted' : output

}, columns= ['Id', 'Predicted'])

submission['Id']=submission['Id'].apply(lambda x: '{0:0>5}'.format(x))

submission.head()

submission.to_csv('sampleSubmission.csv',index=False)

#submission['Predicted'] = output

#submission['Id'] = list(range(0, len(output+1)))

#submission
from IPython.display import FileLink, FileLinks

FileLinks('.') #lists all downloadable files on server
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(submission)