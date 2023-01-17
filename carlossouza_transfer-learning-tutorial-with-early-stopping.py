%%time

!pip install '/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4'
from __future__ import print_function, division



import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import pandas as pd

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

import pretrainedmodels

import pretrainedmodels.utils as utils

from shutil import copyfile

os.environ['TORCH_HOME'] = '/kaggle/working/pretrained-model-weights-pytorch'
print(pretrainedmodels.model_names)
def copy_weights(model_name):

    found = False

    for dirname, _, filenames in os.walk('/kaggle/input/pretrained-model-weights-pytorch'):

        for filename in filenames:

            full_path = os.path.join(dirname, filename)

            if filename.startswith(model_name):

                found = True

                break

        if found:

            break

            

    base_dir = "/kaggle/working/pretrained-model-weights-pytorch/checkpoints"

    os.makedirs(base_dir, exist_ok=True)

    filename = os.path.basename(full_path)

    copyfile(full_path, os.path.join(base_dir, filename))
copy_weights('xception')

model_name = 'xception' # could be fbresnet152 or inceptionresnetv2

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

model.eval()

tf_img = utils.TransformImage(model)
data_dir = '/kaggle/input/pytorchtransferlearningtutorial/hymenoptera_data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

                                          tf_img)

                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,

                                             shuffle=True, num_workers=1)

              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def imshow(inp, model=model, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    inp = model.std * inp + model.mean

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
class EarlyStopping(object):

    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):

        self.mode = mode

        self.min_delta = min_delta

        self.patience = patience

        self.best = None

        self.num_bad_epochs = 0

        self.is_better = None

        self._init_is_better(mode, min_delta, percentage)



        if patience == 0:

            self.is_better = lambda a, b: True

            self.step = lambda a: False



    def step(self, metrics):

        if self.best is None:

            self.best = metrics

            return False



        if np.isnan(metrics):

            return True



        if self.is_better(metrics, self.best):

            self.num_bad_epochs = 0

            self.best = metrics

            print('improvement!')

        else:

            self.num_bad_epochs += 1

            print(f'no improvement, bad_epochs counter: {self.num_bad_epochs}')



        if self.num_bad_epochs >= self.patience:

            return True



        return False



    def _init_is_better(self, mode, min_delta, percentage):

        if mode not in {'min', 'max'}:

            raise ValueError('mode ' + mode + ' is unknown!')

        if not percentage:

            if mode == 'min':

                self.is_better = lambda a, best: a < best - min_delta

            if mode == 'max':

                self.is_better = lambda a, best: a > best + min_delta

        else:

            if mode == 'min':

                self.is_better = lambda a, best: a < best - (

                            best * min_delta / 100)

            if mode == 'max':

                self.is_better = lambda a, best: a > best + (

                            best * min_delta / 100)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=5):

    

    es = EarlyStopping(patience=patience)

    terminate_training = False

    

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

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

            if phase == 'train':

                scheduler.step()



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

                

            if phase == 'val' and es.step(epoch_loss):

                terminate_training = True

                print('early stop criterion is met, we can stop now')

                break

    

        print()

        if terminate_training:

            break



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
model_ft = model

num_ftrs = model_ft.last_linear.in_features

model_ft.last_linear = nn.Linear(num_ftrs, 2)



model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
%%time

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=100)
visualize_model(model_ft)