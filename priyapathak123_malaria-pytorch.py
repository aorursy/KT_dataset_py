# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function, division

import os

import torch

import pandas as pd

from skimage import io, transform

import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils





import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

# Ignore warnings

import warnings

warnings.filterwarnings("ignore")



plt.ion()   # interactive mode
from torchvision import transforms, datasets



data_transform = transforms.Compose([

        transforms.RandomSizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    ])

dataset = datasets.ImageFolder(root='../input/cell-images-for-detecting-malaria/cell_images/cell_images',

                                           transform=data_transform)

dataset_loader = torch.utils.data.DataLoader(dataset,

                                             batch_size=4, shuffle=True,

                                             num_workers=4)
!pip install split-folders

'''Train and Test split on a dataset with directories'''

import split_folders

orig_path = '../input/cell-images-for-detecting-malaria/cell_images/cell_images'

output_path = '../output'

split_folders.ratio(orig_path, output=output_path, seed=1, ratio=(.8, .2))

'''Preview the split'''

data_dir = '../output'

print(os.listdir(data_dir))
'''Creating train and test paths'''

train = data_dir+'/train/'

test = data_dir+'/val/'
train
def load_dataset():

    data_path = '../output/train/'

    train_dataset = torchvision.datasets.ImageFolder(

        root=data_path,

        transform=torchvision.transforms.ToTensor()

    )

    train_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=64,

        num_workers=0,

        shuffle=True

    )

    return train_loader



#for batch_idx, (data, target) in enumerate(load_dataset()):

    #train network
test
# Data augmentation and normalization for training

# Just normalization for validation

data_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'val': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}



data_dir = '../output'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),

                                          data_transforms[x])

                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,

                                             shuffle=True, num_workers=4)

              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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



imshow(out, title=[class_names[x] for x in classes])
#training the model

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



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
#Visualizing the model predictions

#Generic function to display predictions for a few images



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
from __future__ import print_function, division



import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy
#Finetuning the convnet

#Load a pretrained model and reset final fully connected layer.



model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc = nn.Linear(num_ftrs, 2)



model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#Train and evaluate

#It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.



model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

                       num_epochs=25)
visualize_model(model_ft)