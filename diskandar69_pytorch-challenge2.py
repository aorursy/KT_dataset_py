import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import json
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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
device = torch.cuda.is_available()
device, torch.__version__

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
print(train_on_gpu)
data_dir = '../input/flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
torch.cuda.is_available()
train_dir
# TODO: Define transforms for the training data and testing data
image_transforms = { 'train': transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]),
                  
                   'valid': transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(train_dir, image_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, image_transforms['valid'])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=0)
valid_loader = DataLoader(valid_data, batch_size=16,num_workers=0)
image_transforms['train']
import json
class_names = train_data.classes
with open('../input/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def imshow(inp, title=None):
    """Imshow for Tensor."""
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    inp = inp.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs,nrow=8)

imshow(out,title=[cat_to_name[str(x.item())] for x in classes])
# TODO: Build and train your network
model = models.resnet18(pretrained=True)
# Freeze the weights
for param in model.parameters():
    param.requires_grad = False
    
#define untrained feed-forward network as a classifier
num_inp = model.fc.in_features
model.fc = nn.Linear(num_inp, 102)
if train_on_gpu:
    model.cuda()
#TO UNFREEZE THE WEIGHTS FOR THE SECOND TIME TRAINING AND DON'T FORGET TO CHANGE THE OPTIMIZER TOO!
for param in model.parameters():
    param.requires_grad = True
    
# train the WHOLE part instead of just the classifier!, THIS IS THE SECOND ITERATION OF THE TRAINING AND REDUCE THE LEARNING RATE TOO!
optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
models.resnet18()
criterion = torch.nn.CrossEntropyLoss()
#only train the classifier part, the features part are frozen
optimizer = optim.SGD(model.fc.parameters(), lr=0.002, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
len(train_loader), len(valid_loader), len(train_data), len(valid_data)
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
                dataloader = train_loader
                dataset_size = len(train_data)
                model.train()  # Set model to training mode
            else:
                dataloader = valid_loader
                dataset_size = len(valid_data)
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

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
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs = 20)
# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

#WITH ACCURACY METRICS
epochs = 10
steps = 0
test_loss_min = np.Inf # track change in test/validation loss
print_every = 10

for epoch in range(epochs):
    running_loss = 0
    test_loss = 0
    model.train()
    
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            # calculate average losses
            running_loss = running_loss/len(train_loader)
            test_loss = test_loss/len(valid_loader.dataset)
            
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss:.3f}.. "
                  f"Test accuracy: {accuracy:.3f}")
            
            # save model if validation loss has decreased
            if test_loss <= test_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                test_loss_min,
                test_loss))
                torch.save(model.state_dict(), 'model_cifar.pt')
                test_loss_min = test_loss
# TODO: Save the checkpoint 
# TODO: Write a function that loads a checkpoint and rebuilds the model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
# TODO: Display an image along with the top 5 classes