import os

print(os.listdir("../input/images/Images"))

path = "../input/images/Images"
import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.datasets import load_files

import torch.optim as optim

import os

import numpy as np

import time

from PIL import Image

import torchvision

from torchvision.utils import make_grid

from torchvision import datasets,transforms, models

from torch.utils.data import Dataset

from torchvision.datasets import ImageFolder

from torch.autograd import Variable

import matplotlib.pyplot as plt

import copy

from glob import glob
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import cv2



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models
files_training = glob(os.path.join(path, '*/*.jpg'))

num_images = len(files_training)

print('Number of images in Training file:', num_images)
min_images = 1000

im_cnt = []

class_names = []

print('{:29s}'.format('class'), end='')

print('Count:')

print('-' * 35)

for folder in os.listdir(os.path.join(path)):

    folder_num = len(os.listdir(os.path.join(path,folder)))

    im_cnt.append(folder_num)

    class_names.append(folder)

    print('{:30s}'.format(folder[10:]), end=' ')

    print(folder_num)

    if (folder_num < min_images):

        min_images = folder_num

        folder_name = folder

        

num_classes = len(class_names)

print("\nMinumum imgages per category:", min_images, 'Category:', folder[10:])    

print('Average number of Images per Category: {:.0f}'.format(np.array(im_cnt).mean()))

print('Total number of classes: {}'.format(num_classes))
tensor_transform = transforms.Compose([

    transforms.ToTensor()

])



all_data = ImageFolder(os.path.join(path), tensor_transform)



data_loader = torch.utils.data.DataLoader(all_data, batch_size=1, shuffle=True)
#pop_mean = []

#pop_std = []



#for i, data in enumerate(data_loader, 0):

#    numpy_image = data[0].numpy()

#    

#    batch_mean = np.mean(numpy_image, axis=(0,2,3))

#    batch_std = np.std(numpy_image, axis=(0,2,3))

    

#    pop_mean.append(batch_mean)

#    pop_std.append(batch_std)



#pop_mean = np.array(pop_mean).mean(axis=0)

#pop_std = np.array(pop_std).mean(axis=0)
#print(pop_mean)

#print(pop_std)
np.random.seed(123)

shuffle = np.random.permutation(num_images)
split_val = int(num_images * 0.2)

print('Total number of images:', num_images)

print('Number of valid images after split:',len(shuffle[:split_val]))

print('Number of train images after split:',len(shuffle[split_val:]))
class TrainDataset(Dataset):

    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):

        self.shuffle = shuffle

        self.class_names = class_names

        self.split_val = split_val

        self.data = np.array([files[i] for i in shuffle[split_val:]])

        self.transform=transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img = Image.open(self.data[idx]).convert('RGB')

        name = self.data[idx].split('/')[-2]

        y = self.class_names.index(name)

        img = self.transform(img)

            

        return img, y



class ValidDataset(Dataset):

    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):

        self.shuffle = shuffle

        self.class_names = class_names

        self.split_val = split_val

        self.data = np.array([files[i] for i in shuffle[:split_val]])

        self.transform=transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img = Image.open(self.data[idx]).convert('RGB')

        name = self.data[idx].split('/')[-2]

        y = self.class_names.index(name)

        img = self.transform(img)

            

        return img, y
num_classes = len(class_names)
data_transforms = {

    'train': transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225]) # These were the mean and standard deviations that we calculated earlier.

    ]),

    'valid': transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], 

                             [0.229, 0.224, 0.225]) # These were the mean and standard deviations that we calculated earlier.

    ])

}



train_dataset = TrainDataset(files_training, shuffle, split_val, class_names, data_transforms['train'])

valid_dataset = ValidDataset(files_training, shuffle, split_val, class_names, data_transforms['valid'])





train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True)

dataloaders = {'train': train_loader,

              'valid': valid_loader}

dataset_sizes = {

    'train': len(train_dataset),

    'valid': len(valid_dataset)

}
pop_mean = [0.4761387,  0.45182642, 0.39102373]

pop_std = [0.23377682, 0.22895287, 0.22751313]

# This allows us to see some of the fruits in each of the datasets 

def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    inp = pop_std * inp + pop_mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.01)  # pause a bit so that plots are updated
# Here we are just checking out the next batch of images from the train_loader, and below I print the class names. 

inputs, classes = next(iter(train_loader))

out = make_grid(inputs)



cats = ['' for x in range(len(classes))]

for i in range(len(classes)):

    cats[i] = class_names[classes[i].item()]

    

imshow(out)

print(cats)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim import lr_scheduler

model_ft = models.resnet50(pretrained=True)

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 120)



model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
inputs, classes = next(iter(dataloaders['train']))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs)



imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

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

            if phase == 'valid' and epoch_acc > best_acc:

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
def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()



    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloaders['valid']):

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
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=12)

visualize_model(model_ft)