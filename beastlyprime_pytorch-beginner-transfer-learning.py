from __future__ import print_function, division



import torch

import torch.nn as nn



import time

import copy
torch.cuda.is_available()
# Another way to copy the pretrained models to the cache directory (~/.torch/models) where PyTorch is looking for them.

# From https://www.kaggle.com/pvlima/use-pretrained-pytorch-models#Transfer-learning-in-kernels-with-PyTorch

import os

from os import listdir, makedirs, getcwd, remove

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.torch'))



if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)

    

!cp ../input/pretrained-pytorch-models/* ~/.torch/models/

!ls ~/.torch/models
import torchvision

from torchvision import datasets, transforms



data_transforms = {

    'train': transforms.Compose([   # Here we do not make data augmentations

        transforms.Resize(325),

        transforms.CenterCrop(299), # Note that we want to use Inception v3, it requires this size of images

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # We can simply use this parameter

    ]),

    'val': transforms.Compose([

        transforms.Resize(325),

        transforms.CenterCrop(299),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}



data_dir = "../input/dogs-vs-cats/dataset/dataset"



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, name),

                                          data_transforms[x])

                  for x, name in [['train', "training_set"], ['val', "test_set"]]}



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,

                                             shuffle=True, num_workers=4)

              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes



# See some statistics

print(dataloaders)

len(dataloaders['train'])
import numpy as np

import matplotlib.pyplot as plt

plt.ion()   # interactive mode



def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

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



imshow(out)



# imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, num_epochs=2, is_inception=False):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

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

                    # Get model outputs and calculate loss

                    # Special case for inception because in training it has an auxiliary output. In train

                    # mode we calculate the loss by summing the final output and the auxiliary output

                    # but in testing we only consider the final output.

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
import torchvision.models as models

import torch.optim as optim



model_ft = models.inception_v3(pretrained=True)

for param in model_ft.parameters():

    param.requires_grad = False



# Parameters of newly constructed modules have requires_grad=True by default

# Handle the auxilary net

num_ftrs = model_ft.AuxLogits.fc.in_features

model_ft.AuxLogits.fc = nn.Linear(num_ftrs, 2)

# Handle the primary net

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 2)



# print(model_ft)
from torch.optim import lr_scheduler



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model_ft = model_ft.to(device)



# Observe that only parameters of final layer are being optimized as

# opposed to before.

params_to_update = []

for name,param in model_ft.named_parameters():

    if param.requires_grad == True:

        params_to_update.append(param)



optimizer_conv = optim.SGD(params_to_update, lr=0.001, momentum=0.9)



criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every epoch

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_conv,

                         exp_lr_scheduler, num_epochs=2, is_inception=True) # As an example, only show the results of 2 epoch
visualize_model(model_ft)



plt.ioff()

plt.show()
# Note that this way of copying files will generate outputs.

"""

%mkdir -p data/train

%mkdir -p data/val

%mkdir -p /tmp/.torch/models



%cp -r ../input/dogs-vs-cats/dataset/dataset/training_set/* data/train

%cp -r ../input/dogs-vs-cats/dataset/dataset/test_set/* data/val

%cp -r ../input/pretrained-pytorch-models/* /tmp/.torch/models



import os

print(os.listdir("data"))

print(os.listdir("/tmp/.torch/models"))

"""