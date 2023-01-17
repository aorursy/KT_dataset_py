# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import numpy as np

# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
  print('CUDA is not available. Training on CPU...')
else:
  print('CUDA is available. Training on GPU')
#Load the data 

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

# Number of subprocesses to use for data loading
num_workers = 0

# Number of samples per batch to load
batch_size = 32

# Percentage of training set to use for validation
valid_size = 0.2


# Loading train and test data
train_data = datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_train/seg_train',
                                 transform = transforms.ToTensor())
test_data = datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_test/seg_test',
                                transform = transforms.ToTensor())

import numpy as np

batch_size=32
img_dimensions = 224

# Normalize to the ImageNet mean and standard deviation
# Could calculate it for the cats/dogs data set, but the ImageNet
# values give acceptable results here.
transform_train = transforms.Compose([
    transforms.Resize((img_dimensions, img_dimensions)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

transform_test = transforms.Compose([
    transforms.Resize((img_dimensions,img_dimensions)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])


# Rewrite  train and test data
train_data = datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_train/seg_train',
                                 transform = transform_train)
test_data = datasets.ImageFolder('/kaggle/input/intel-image-classification/seg_test/seg_test',
                                transform = transform_test)

# Create validation set
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Create dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                         sampler = train_sampler,
                                         num_workers = num_workers)
validloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                         sampler = valid_sampler,
                                         num_workers = num_workers)
testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                        num_workers = num_workers)

# Get the classes
import pathlib
root = pathlib.Path('/kaggle/input/intel-image-classification/seg_train/seg_train/')
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)
import matplotlib.pyplot as plt
%matplotlib inline

def imshow(img):
  '''
  Function to un-normalize and display an image
  '''
  img = img/2 + 0.5 # un-normalize
  plt.imshow(np.transpose(img, (1, 2, 0))) # convert from tensor image
  
# Get a batch of training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# Plot the images from the batch, along with corresponding labels
fig = plt.figure(figsize = (25, 4))

# Display 20 images
for idx in np.arange(20):
  ax = fig.add_subplot(2, 20/2, idx+1, xticks = [], yticks = [])
  imshow(images[idx])
  ax.set_title(classes[labels[idx]])
print(images.shape)

print(f'Num training images: {len(trainloader.dataset)}')
print(f'Num validation images: {len(validloader.dataset)}')
print(f'Num test images: {len(testloader.dataset)}')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model_resnet34 = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
model_resnet50 = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
# Freeze all params except the BatchNorm layers, as here they are trained to the
# mean and standard deviation of ImageNet and we may lose some signal
for name, param in model_resnet18.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
        
for name, param in model_resnet34.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
        
for name, param in model_resnet50.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
# Replace the classifier
num_classes = 6

num_ftrs1 = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs1, 6)

num_ftrs2 = model_resnet34.fc.in_features
model_resnet34.fc = nn.Linear(num_ftrs2, 6)

num_ftrs = model_resnet50.fc.in_features
model_resnet50.fc = nn.Linear(num_ftrs, 6)


#The old 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, loss_fn, trainloader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(trainloader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
                        
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))

def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('correct: {:d}  total: {:d}'.format(correct, total))
    print('accuracy = {:f}'.format(correct / total))
model_resnet18.to(device)
optimizer = optim.Adam(model_resnet18.parameters(), lr=0.001)
train(model_resnet18, optimizer, torch.nn.CrossEntropyLoss(), trainloader, validloader, epochs=20, device=device)
test_model(model_resnet18)

model_resnet34.to(device)
optimizer = optim.Adam(model_resnet34.parameters(), lr=0.001)
train(model_resnet34, optimizer, torch.nn.CrossEntropyLoss(), trainloader, validloader, epochs=10, device=device)

test_model(model_resnet34)
model_resnet50.to(device)
optimizer = optim.Adam(model_resnet50.parameters(), lr=0.001)
train(model_resnet50, optimizer, torch.nn.CrossEntropyLoss(), trainloader, validloader, epochs=10, device=device)
test_model(model_resnet50)
torch.save(model_resnet18.state_dict(), "./model_resnet18.pth")
torch.save(model_resnet34.state_dict(), "./model_resnet34.pth")
torch.save(model_resnet50.state_dict(), "./model_resnet50.pth")
# Remember that you must call model.eval() to set dropout and batch normalization layers to
# evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

resnet18 = torch.hub.load('pytorch/vision', 'resnet18')
resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
resnet18.load_state_dict(torch.load('./model_resnet18.pth'))
resnet18.eval()

resnet34 = torch.hub.load('pytorch/vision', 'resnet34')
resnet34.fc = nn.Sequential(nn.Linear(resnet34.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
resnet34.load_state_dict(torch.load('./model_resnet34.pth'))
resnet34.eval()

resnet50 = torch.hub.load('pytorch/vision', 'resnet50')
resnet50.fc = nn.Sequential(nn.Linear(resnet50.fc.in_features,512),nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
resnet50.load_state_dict(torch.load('./model_resnet50.pth'))
resnet50.eval()
# Test against the average of each prediction from the two models# Test against the average of each prediction from the two models
models_ensemble = [resnet18.to(device), resnet34.to(device),resnet50.to(device)]
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        predictions = [i(images).data for i in models_ensemble]
        avg_predictions = torch.mean(torch.stack(predictions), dim=0)
        _, predicted = torch.max(avg_predictions, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('accuracy = {:f}'.format(correct / total))
print('correct: {:d}  total: {:d}'.format(correct, total))