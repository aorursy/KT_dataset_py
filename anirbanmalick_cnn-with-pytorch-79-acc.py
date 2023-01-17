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

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
!pip install py7zr

from py7zr import unpack_7zarchive
import shutil
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
shutil.unpack_archive('/kaggle/input/cifar-10/train.7z', '/kaggle/working')
raw_data_dir = '/kaggle/working/train/'
len(os.listdir('/kaggle/working/train/'))
import matplotlib.pyplot as plt
label_data = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
print(label_data.head())
pd.DataFrame(label_data.label.value_counts())

targets = np.array(label_data.label)
from sklearn.model_selection import train_test_split
train_idx, test_idx= train_test_split(
label_data,
test_size=0.2,
shuffle=True,
stratify=targets)
print(train_idx.label.value_counts())
#train_idx,valid_idx = list(train_idx.id),list(valid_idx.id)
train_idx = train_idx.reset_index(drop=True)
test_idx = test_idx.reset_index(drop=True)
from tqdm import tqdm
targets = list(set(targets))
work_dir = '/kaggle/working/'
train_path_name = work_dir+'train_classified/'
test_path_name = work_dir+'test_classified/'

train_target_folders = [train_path_name+t for t in targets]
test_target_folders = [test_path_name+t for t in targets]

make_folders = [train_path_name,test_path_name] + train_target_folders + test_target_folders

def copy_data(label_data,destination):
    for _,row in tqdm(label_data.iterrows()):
        if not os.path.exists(destination+row['label']+'/'+str(row['id'])+'.png'):
            shutil.copy(raw_data_dir+str(row['id'])+'.png',destination+row['label'])
    return

for path in tqdm(make_folders):
    if not os.path.exists(path):
        os.makedirs(path)
#     print('Folders Created')
        
copy_data(label_data=train_idx,destination=train_path_name)
copy_data(label_data=test_idx,destination=test_path_name)
import torch
from torchvision import datasets
import torchvision.transforms as transforms


# convert data to a normalized torch.FloatTensor
train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    #transforms.
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


train_data = datasets.ImageFolder(train_path_name, transform=train_transform)
test_data = datasets.ImageFolder(test_path_name, transform=test_transform)
train_data,test_data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50,shuffle=True)
valid_loader = torch.utils.data.DataLoader(test_data, batch_size=50,shuffle=True)
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)

        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        #FC Network
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,32)
        self.fc4 = nn.Linear(32,10)
        
        
        #Dropout
        self.dropout_fc = nn.Dropout(0.2)
        

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.leaky_relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(F.leaky_relu(self.conv2(x)))
#         print(x.shape)
        #x = self.dropout_cnn(x)
        x = self.pool(F.leaky_relu(self.conv3(x)))

#         print(x.shape)

        #print(x.shape)
        
        #Flatten
        x = x.view(-1, 128 * 4 * 4)
        #x = self.dropout_fc(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout_fc(x)
        x = self.fc4(x)
        
        return x

# create a complete CNN
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model
n_epochs = 75

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
#     # save model if validation loss has decreased
#     if valid_loss <= valid_loss_min:
#         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#         valid_loss_min,
#         valid_loss))
#         torch.save(model.state_dict(), 'model_cifar.pt')
#         valid_loss_min = valid_loss
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
batch_size = 50
classes = targets
model.eval()
# iterate over test data
for data, target in valid_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(valid_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))