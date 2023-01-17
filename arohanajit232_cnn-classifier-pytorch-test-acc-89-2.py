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
!pip install torchsummary
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import shutil
train_on_gpu = torch.cuda.is_available()
print(train_on_gpu)
for i in os.listdir('../input/ckplus/CK+48'):
    a = '../input/ckplus/CK+48/'+i
    print(i,len(os.listdir(a)))
data_transforms = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
data = ImageFolder('../input/ckplus/CK+48/',transform=data_transforms)
test_size = 0.2
print(len(data))
class_labels = list(data.class_to_idx.keys())
print(class_labels)
num_train = len(data)
indices = list(range(num_train))
np.random.shuffle(indices)
test_split = int(np.floor(test_size * num_train))
test_idx, train_idx = indices[:test_split], indices[test_split:]

print(len(test_idx), len(train_idx))

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
dataset_sizes = {
    'train' : len(train_idx),
    'test' : len(test_idx)
}

loaders = {
    'train': torch.utils.data.DataLoader(data, batch_size=8, sampler=train_sampler),
    'test': torch.utils.data.DataLoader(data, batch_size=8, sampler=test_sampler)
}
import matplotlib.pyplot as plt
%matplotlib inline


def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0)))

dataiter = iter(loaders['train'])
images, labels = dataiter.next()
print(images.shape,labels.shape)
images = images.numpy() 
fig = plt.figure(figsize=(25, 16))

for idx in np.arange(8):
    ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(class_labels[int(labels[idx])],fontsize=20,color='white')
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2)
        )
        
            
        self.fc1 = nn.Linear(32*4*4, 512)
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 7)
        
    def forward(self, x):
        out = self.block(x)
        out = self.block2(out)
        out = self.block2(out)
        out = out.view(out.size(0), -1)   # flatten out a input for Dense Layer
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
def graph_loss(train_loss,test_loss):
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(train_loss)
    ax.plot(test_loss)
    ax.legend(['Train Loss','Test Loss'])
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    train_loss,test_loss = [],[]
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in loaders[phase]:
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
            
            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                test_loss.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    graph_loss(train_loss, test_loss)
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            fig = plt.figure(figsize=(25, 16))

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(2, num_images//2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_labels[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_ft = Net()
model_ft = model_ft.to(device)
print(summary(model_ft,(3,50,50)))
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=100)
visualize_model(model_ft)

x = torch.rand(1,3,50,50).cpu()
torch.jit.save(torch.jit.trace(model_ft.cpu(), (x)), "model.pth")
