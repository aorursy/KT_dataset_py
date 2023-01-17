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
path = "../input/simpsons_dataset"
files_training = glob(os.path.join(path,'simpsons_dataset', '*/*.jpg'))

num_images = len(files_training)

print('Number of images in Training file:', num_images)
min_images = 1000

im_cnt = []

class_names = []

print('{:18s}'.format('class'), end='')

print('Count:')

print('-' * 24)

for folder in os.listdir(os.path.join(path, 'simpsons_dataset')):

    folder_num = len(os.listdir(os.path.join(path,'simpsons_dataset',folder)))

    im_cnt.append(folder_num)

    class_names.append(folder)

    print('{:20s}'.format(folder), end=' ')

    print(folder_num)

    if (folder_num < min_images):

        min_images = folder_num

        folder_name = folder

        

num_classes = len(class_names)

print("\nMinumum imgages per category:", min_images, 'Category:', folder)    

print('Average number of Images per Category: {:.0f}'.format(np.array(im_cnt).mean()))

print('Total number of classes: {}'.format(num_classes))
tensor_transform = transforms.Compose([

    transforms.ToTensor()

])



all_data = ImageFolder(os.path.join(path, 'simpsons_dataset'), tensor_transform)
data_loader = torch.utils.data.DataLoader(all_data, batch_size=1, shuffle=True)
pop_mean = []

pop_std = []



for i, data in enumerate(data_loader, 0):

    numpy_image = data[0].numpy()

    

    batch_mean = np.mean(numpy_image, axis=(0,2,3))

    batch_std = np.std(numpy_image, axis=(0,2,3))

    

    pop_mean.append(batch_mean)

    pop_std.append(batch_std)



pop_mean = np.array(pop_mean).mean(axis=0)

pop_std = np.array(pop_std).mean(axis=0)
print(pop_mean)

print(pop_std)
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

        img = Image.open(self.data[idx])

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

        img = Image.open(self.data[idx])

        name = self.data[idx].split('/')[-2]

        y = self.class_names.index(name)

        img = self.transform(img)

            

        return img, y

    

class TestDataset(Dataset):

    def __init__(self, path, class_names, transform=transforms.ToTensor()):

        self.class_names = class_names

        self.data = np.array(glob(os.path.join(path, '*/*.jpg')))

        self.transform=transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img = Image.open(self.data[idx])

        name = self.data[idx].split('/')[-2]

        y = self.class_names.index(name)

        img = self.transform(img)

            

        return img

    
class_names
data_transforms = {

    'train': transforms.Compose([

        transforms.Resize((60, 60)),

        transforms.ToTensor(),

        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.

    ]),

    'test': transforms.Compose([

        transforms.Resize((60, 60)),

        transforms.ToTensor(),

        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.

    ]),

    'valid': transforms.Compose([

        transforms.Resize((60, 60)),

        transforms.ToTensor(),

        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.

    ])

}



train_dataset = TrainDataset(files_training, shuffle, split_val, class_names, data_transforms['train'])

valid_dataset = ValidDataset(files_training, shuffle, split_val, class_names, data_transforms['valid'])

test_dataset = TestDataset('../input/kaggle_simpson_testset/', class_names, transform=data_transforms['test'])



train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
### These are just used for convinience purposes



dataloaders = {'train': train_loader,

              'valid': valid_loader,

              'test': test_loader}

dataset_sizes = {

    'train': len(train_dataset),

    'valid': len(valid_dataset),

    'test': len(test_dataset)

}



len(test_dataset)
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
class Net(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # Conv 1

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.relu1 = nn.ReLU()

        

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Conv 2

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.relu2 = nn.ReLU()

        

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        

        

        # Fully conected 1

        self.fc1 = nn.Linear(32 * 15 * 15, 42)

        

    def forward(self, x):

        # Convolution 1

        out = self.cnn1(x)

        out = self.relu1(out)



        # Max pool 1

        out = self.maxpool1(out)



        # Convolution 2 

        out = self.cnn2(out)

        out = self.relu2(out)



        # Max pool 2 

        out = self.maxpool2(out)



        # Resize

        # Original size: (100, 32, 7, 7)

        # out.size(0): 100

        # New out size: (100, 32*7*7)

        out = out.view(out.size(0), -1)



        # Linear function (readout)

        out = self.fc1(out)



        return out



        return F.log_softmax(x, dim=1) 
model = Net(num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net(num_classes)

model.to(device)



optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

exp_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
print(model.parameters())



print(len(list(model.parameters())))



# Convolution 1: 16 Kernels

print(list(model.parameters())[0].size())



# Convolution 1 Bias: 16 Kernels

print(list(model.parameters())[1].size())



# Convolution 2: 32 Kernels with depth = 16

print(list(model.parameters())[2].size())



# Convolution 2 Bias: 32 Kernels with depth = 16

print(list(model.parameters())[3].size())



# Fully Connected Layer 1

print(list(model.parameters())[4].size())



# Fully Connected Layer Bias

print(list(model.parameters())[5].size())
def fit(model, criterion, optimizer, scheduler, num_epochs=30):

    since = time.time() # allows us to keep track of how long it took

    

    best_acc = 0.0 # allows us to store the best_acc rate (for validation stage)

    

    # Loop through the data-set num_epochs times.

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 15)

        

        for phase in ['train', 'valid']:

            if phase == 'train':

                scheduler.step()

                model.train() # This sets the model to training mode

            else:

                model.eval() # this sets the model to evaluation mode 

                

            running_loss = 0.0

            running_corrects = 0

            

            # using the dataloaders to load data in batches

            for inputs, labels in dataloaders[phase]:

                # putting the inputs and labels on cuda (gpu)

                inputs = inputs.to(device) 

                labels = labels.to(device)

                

                # zero the gradient

                optimizer.zero_grad()

                

                # if training phase, allow calculating the gradient, but don't allow otherwise

                with torch.set_grad_enabled(phase == 'train'):

                    # get outputs and predictions

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    

                    loss = criterion(outputs, labels) # get value of loss function with the current weights 

                    

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                        

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

                

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            

            # keep track of the best weights for the validation dataset 

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

                

        print()

        

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best validation Acc: {:4f}'.format(best_acc))

    

    model.load_state_dict(best_model_wts)

    return model
model = fit(model, criterion, optimizer, exp_scheduler, num_epochs=12)
from torchvision import transforms

transform = transforms.Compose([            #[1]

 transforms.Resize(70),                    #[2]

 transforms.CenterCrop(60),                #[3]

 transforms.ToTensor(),                     #[4]

 transforms.Normalize(                      #[5]

 mean=[0.485, 0.456, 0.406],                #[6]

 std=[0.229, 0.224, 0.225]                  #[7]

 )])



   
img = Image.open('../input/kaggle_simpson_testset/kaggle_simpson_testset/abraham_grampa_simpson_0.jpg')

img
img_t = transform(img)

batch_t = torch.unsqueeze(img_t, 0)

batch_t = batch_t.to(device)
model.eval()

out = model(batch_t)

print(out.shape)
_, index = torch.max(out, 1)

 

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

 

print(percentage[index[0]].item())
