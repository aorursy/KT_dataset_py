import torch

import torchvision

from torchvision import transforms, models

import torch.nn as nn

import torch.nn.functional as F

from torchvision import datasets

import numpy as np



import os

import copy

import time

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser





import matplotlib.pyplot as plt
cache_dir = expanduser(join('~', '.torch'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)
!cp ../input/resnet34/* ~/.torch/models/
!ls ~/.torch/models
data_transforms = {

    'seg_train': transforms.Compose([

         transforms.RandomResizedCrop(224),

         transforms.RandomHorizontalFlip(),

         transforms.ToTensor(),

         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'seg_test': transforms.Compose([

         transforms.Resize(256),

         transforms.CenterCrop(224),

         transforms.ToTensor(),

         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

}
image_train = {x: torchvision.datasets.ImageFolder('../input/intel-image-classification/seg_train/seg_train', data_transforms[x]) for x in ['seg_train']}

data_train = {x: torch.utils.data.DataLoader(image_train[x], batch_size=16,

                                              shuffle=True, num_workers=4) for x in ['seg_train']}



dataset_train = {x: len(image_train[x]) for x in ['seg_train']}

class_names = image_train['seg_train'].classes



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_val = {x: torchvision.datasets.ImageFolder('../input/intel-image-classification/seg_test/seg_test', data_transforms[x]) for x in ['seg_test']}

data_val = {x: torch.utils.data.DataLoader(image_val[x], batch_size=16,

                                              shuffle=True, num_workers=4) for x in ['seg_test']}



dataset_val = {x: len(image_val[x]) for x in ['seg_test']}
class Img_clf(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 512, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128*28*28, 64)

        self.fc2 = nn.Linear(64, 6)

        

    def forward(self, x):

        out = F.relu(F.max_pool2d(self.conv1(x),2))

        out = F.relu(F.max_pool2d(self.conv2(out),2))

        out = F.relu(F.max_pool2d(self.conv3(out),2))

        out = out.view(-1, 128*28*28)

        out = F.relu(self.fc1(out))

        out = self.fc2(out)

        

        return out
model = Img_clf()

model.to(device) #Moving to Cuda if available
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())
def train_model(model, criterion, optimizer, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for data in [data_train, data_val]:

            phase = list(data.keys())[0]

            if phase == 'seg_train':

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in data[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'seg_train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'seg_train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

            

            if phase == 'seg_train':

                epoch_loss = running_loss / dataset_train[phase]

                epoch_acc = running_corrects.double() / dataset_train[phase]

            else:

                epoch_loss = running_loss / dataset_val[phase]

                epoch_acc = running_corrects.double() / dataset_val[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'seg_test' and epoch_acc > best_acc:

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
trained_model = train_model(model, criterion, optimizer, num_epochs=5)
transfer_model = models.resnet34(pretrained=True)

for name, param in transfer_model.named_parameters():

    if ('bn' not in name):

        param.requires_grad = False
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features, 500), nn.ReLU(), nn.Dropout(), nn.Linear(500, 6))
### Pretrained Model

optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.01)

transfer_model.to(device)

trained_tl_model = train_model(transfer_model, criterion, optimizer,

                       num_epochs=5)
data_test_transform = {

    'seg_pred': transforms.Compose([

         transforms.Resize(256),

         transforms.CenterCrop(224),

         transforms.ToTensor(),

         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

}
image_test = {x: torchvision.datasets.ImageFolder("../input/intel-image-classification/seg_pred", data_test_transform[x]) for x in ['seg_pred']}

data_test = {x: torch.utils.data.DataLoader(image_test[x], batch_size=8,

                                              shuffle=True, num_workers=4) for x in ['seg_pred']}



dataset_sizes = {x: len(image_test[x]) for x in ['seg_pred']}
model.eval()   # Set model to evaluate mode

for inputs, labels in data_test["seg_pred"]:

    inputs = inputs.to(device)

    outputs = trained_tl_model(inputs)

    _, preds = torch.max(outputs, 1)

    print(preds)

    break
trained_model.to('cpu')

def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.detach().numpy()

    inp = inp.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)



    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated





# Get a batch of training data

inputs, classes = next(iter(data_test['seg_pred']))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs)

#inputs = inputs#.to(device)

outputs = trained_model(inputs)

_, preds = torch.max(outputs, 1)



plt.figure(figsize=(16,16))

imshow(out, title=[class_names[x] for x in preds])
trained_tl_model.to('cpu')

def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.detach().numpy()

    inp = inp.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)



    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated





# Get a batch of training data

inputs, classes = next(iter(data_test['seg_pred']))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs)

#inputs = inputs#.to(device)

outputs = trained_tl_model(inputs)

_, preds = torch.max(outputs, 1)



plt.figure(figsize=(16,16))

imshow(out, title=[class_names[x] for x in preds])