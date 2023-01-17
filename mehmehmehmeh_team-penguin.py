import glob
import imghdr
import os

src_dir = "Training%20Images"
dst_dir = "Evaluating%20Images"


FOLDER_NAME = ["BabyBibs","BabyHat","BabyPants","BabyShirt","PackageFart","womanshirtsleeve","womencasualshoes","womenchiffontop","womendollshoes",\
               "womenknittedtop","womenlazyshoes","womenlongsleevetop","womenpeashoes","womenplussizedtop","womenpointedflatshoes","womensleevelesstop",\
               "womenstripedtop","wrapsnslings"]

for folder in FOLDER_NAME:

    if not os.path.exists(os.path.join(dst_dir, folder)):
        os.makedirs(os.path.join(dst_dir, folder))

    files_copy = glob.iglob(os.path.join(src_dir, folder, '*.jpg'))
    for jpgfile in files_copy:
        if imghdr.what(os.path.join(jpgfile)) != 'jpeg':
                os.remove(os.path.join(jpgfile))
        else:
            if '5' in jpgfile:
                shutil.copy(jpgfile, os.path.join(dst_dir, folder))

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import fnmatch
from skimage import io
from PIL import Image

plt.ion() 


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'Training%20Images': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Evaluating%20Images': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

data_dir = 'tf_files'

FOLDER_NAME = ["BabyBibs","BabyHat","BabyPants","BabyShirt","PackageFart","womanshirtsleeve","womencasualshoes","womenchiffontop","womendollshoes","womenknittedtop","womenlazyshoes","womenlongsleevetop","womenpeashoes","womenplussizedtop","womenpointedflatshoes","womensleevelesstop","womenstripedtop","wrapsnslings"]
 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['Training%20Images', 'Evaluating%20Images']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=False, num_workers=4)
              for x in ['Training%20Images', 'Evaluating%20Images']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Training%20Images', 'Evaluating%20Images']}
class_names = image_datasets['Training%20Images'].classes
print(class_names)
use_gpu = torch.cuda.is_available()
print(dataset_sizes)
print(use_gpu)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Training%20Images', 'Evaluating%20Images']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'Training%20Images':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'Evaluating%20Images' and epoch_acc > best_acc:
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

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 18)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=24)

torch.save(model_ft, 'trainedmodel.pty')

import torch
from PIL import Image
import torchvision
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import os
import imghdr

## HELPER VARIABLES
IMGSIZE = 256
CROPSIZE = 224

IMG_FOLDER_NAME = "tf_files/Testing%20Images/Test"

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMGSIZE), 
    transforms.CenterCrop(CROPSIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

MODEL_NAME = "trainedmodel.pty"

USE_GPU = torch.cuda.is_available()

## HELPER FUNCTIONS
def image_loader(image_path):
    image = Image.open(image_path)
    image = IMAGE_TRANSFORMS(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)

    image_num = image_path.split(".")[0]
    image_num = image_num.split("_")[-1]
    return image_num, image


## MAIN FUNCTION

#load the model



model = torch.load(MODEL_NAME)
# opening the file once to prevent delays due to multiple file openings. 
# opening with w+ to replace all data in the file to create a new submission every time.
with open("submission.csv", "w+") as f:
    # adding heading
    f.write('id,category')
    # not doing it in the same line since that would create a new string in the memory, we want to avoid
    # clogging the memory with unnecessary stuff
    f.write('\n')
    for image in os.listdir(IMG_FOLDER_NAME):
        if imghdr.what(os.path.join(IMG_FOLDER_NAME, image)) == 'jpeg':
            image_num, image_data = image_loader(os.path.join(IMG_FOLDER_NAME, image))
            if USE_GPU:
                image_data = image_data.cuda()
            prediction = model(image_data)
            prediction = prediction.data.max(1, keepdim=True)[1]
            # same reason as above for making it more efficient
            f.write(image_num)
            f.write(',')
            f.write(str(prediction).split("\n")[1].strip())
            f.write('\n')
        else: 
            print(image, end='')
            print(" not found")