import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import json
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import torch
import torchvision
from torchvision import models,datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy

#print the pytorch version
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.__version__
#ImageNet Transforms
image_transforms = { 'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]),
                  
                   'valid': transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
fd_path = "../input/flower_data/flower_data"
train_path = "../input/flower_data/flower_data/train"
valid_path = "../input/flower_data/flower_data/valid"
train_data = datasets.ImageFolder(train_path, image_transforms['train'])
valid_data = datasets.ImageFolder(valid_path, image_transforms['valid'])
train_data_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
valid_data_loader = DataLoader(valid_data, batch_size=16, shuffle=True, num_workers=0)
train_data_loader.dataset.imgs[0]
print('train data size:', len(train_data))
print('valid data size:', len(valid_data))
class_names = train_data.classes
with open('../input/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
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
inputs, classes = next(iter(train_data_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
len(train_data), len(valid_data)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            print(phase)
            if phase == 'train':
                dataloader = train_data_loader
                dataset_size = len(train_data)
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = valid_data_loader
                dataset_size = len(valid_data)
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
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

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

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
model_00 = models.resnet152(pretrained=True)
for param in model_00.parameters():
    param.requires_grad = False


num_ftrs = model_00.fc.in_features
model_00.fc = nn.Linear(num_ftrs, 102)

model_00 = model_00.to(device)

criterion = nn.CrossEntropyLoss()

#TO UNFREEZE THE WEIGHTS FOR THE SECOND TIME TRAINING AND DON'T FORGET TO CHANGE THE OPTIMIZER TOO!
for param in model_00.parameters():
    param.requires_grad = True
    
# train the WHOLE part instead of just the classifier!, THIS IS THE SECOND ITERATION OF THE TRAINING AND REDUCE THE LEARNING RATE TOO!
optimizer = optim.SGD(model_00.parameters(), lr=0.0002, momentum=0.9)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_00.fc.parameters(), lr=0.015, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_00 = train_model(model_00, criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 32)
