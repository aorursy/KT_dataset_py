# for numerical analysis
import numpy as np # linear algebra

# to store and process in a dataframe
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualizations, for ploting graphs
import matplotlib.pyplot as plt

# image processing
import matplotlib.image as mpimg

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# advanced ploting
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Image manipulations
from PIL import Image

# Timing utility
from timeit import default_timer as timer

from IPython.core.interactiveshell import InteractiveShell

# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torchvision
from torchvision import transforms, datasets, models

import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d
from torch.nn import Module, Softmax, BatchNorm2d, Dropout

import gc
# file operations

import shutil
import os
from os import walk

# to list files
import glob

print(os.listdir("../input"))
# current working directory
os.getcwd()
# no. of files

def list_files(startpath):
    
    for root, dirs, files in os.walk(startpath):
        
        level = root.replace(startpath, '').count(os.sep)
        
        indent = ' ' * 4 * (level)
        
        print('{}{}'.format(indent, os.path.basename(root)), '-', len(os.listdir(root)))
        
folder = '/kaggle/input'
list_files(folder)
folder = '/kaggle/working'
list_files(folder)
# list of files in the dataset /input/ecg-images/MITBIH_img

os.listdir('../input/ecg-images/MITBIH_img')
# Classes in the data

ECG_list = os.listdir('../input/ecg-images/MITBIH_img')

n_classes = len(ECG_list)

print(f'There are {n_classes} different classes.')
ECG_list
classes = ('S', 'V', 'Q', 'N', 'F')
N_imgs = os.listdir('../input/ecg-images/MITBIH_img/N')
print('# of Normal beats: ',len(N_imgs))

F_imgs = os.listdir('../input/ecg-images/MITBIH_img/F')
print('# of Fusion beats: ',len(F_imgs))

Q_imgs = os.listdir('../input/ecg-images/MITBIH_img/Q')
print('# of Unknown beats: ',len(Q_imgs))

V_imgs = os.listdir('../input/ecg-images/MITBIH_img/V')
print('# of Ventricular ectopic beats: ',len(V_imgs))

S_imgs = os.listdir('../input/ecg-images/MITBIH_img/S')
print('# of Supraventricular ectopic beats: ',len(S_imgs))
#print(N_dir)
print(N_imgs[0])
def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
image = mpimg.imread(os.path.join('../input/ecg-images/MITBIH_img/N', N_imgs[0]))

imshow(image)
print(image.shape)
print(type(image))
# Define a function which will plot several images

def image_shows(folder, number_of_images):
    
    n=number_of_images;
    
    folder_list = os.listdir(folder)
    
    fig, axes = plt.subplots(nrows = 1, ncols=n, figsize=(20, 10))
    
    for i in range(n):
        
        print(os.path.join(folder, folder_list[i]))
        
        image = mpimg.imread(os.path.join(folder, folder_list[i]));
        
        axes[i].imshow(image);
# Examples of N

image_shows(folder = '../input/ecg-images/MITBIH_img/N', number_of_images = 6)
# Examples of S

image_shows(folder = '../input/ecg-images/MITBIH_img/S', number_of_images = 6)
# Examples of Q

image_shows(folder = '../input/ecg-images/MITBIH_img/Q', number_of_images = 6)
# Examples of V

image_shows(folder = '../input/ecg-images/MITBIH_img/V', number_of_images = 6)
# Examples of F

image_shows(folder = '../input/ecg-images/MITBIH_img/F', number_of_images = 6)
import cv2
imgcv = cv2.imread(os.path.join('../input/ecg-images/MITBIH_img/N', N_imgs[0]))
plt.imshow(imgcv)
plt.show();
imgcv.shape
b = imgcv.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0


g = imgcv.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = imgcv.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0
# plot data

fig = plt.figure(figsize=(15,15))

plot_1 = plt.subplot(131)
plot_1.imshow(r);

#plt.subplot(131).imshow(b);

plot_2 = plt.subplot(132, sharex=plot_1, sharey=plot_1)
plt.setp(plot_2.get_yticklabels(), visible=False);
plot_2.imshow(b);

plot_3 = plt.subplot(133, sharex=plot_1, sharey=plot_1)
plt.setp(plot_3.get_yticklabels(), visible=False);
plot_3.imshow(g);

plt.show();
for root, dirs, files in os.walk('/kaggle/'):
    print(root)
os.makedirs('../working/train/N', exist_ok = True)
os.makedirs('../working/train/F', exist_ok = True)
os.makedirs('../working/train/Q', exist_ok = True)
os.makedirs('../working/train/V', exist_ok = True)
os.makedirs('../working/train/S', exist_ok = True)


os.makedirs('../working/test/N', exist_ok = True)
os.makedirs('../working/test/F', exist_ok = True)
os.makedirs('../working/test/Q', exist_ok = True)
os.makedirs('../working/test/V', exist_ok = True)
os.makedirs('../working/test/S', exist_ok = True)
for root, dirs, files in os.walk('/kaggle/'):
    print(root)
for file_name in F_imgs[0:round(len(F_imgs)/5)]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/F', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/test/F');
        
        
for file_name in F_imgs[round(len(F_imgs)/5):]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/F', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/train/F');

print("done")
# N_imgs = os.listdir('../input/ecg-images/MITBIH_img/N')

for file_name in N_imgs[0:round(len(N_imgs)/5)]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/N', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/test/N');
        
for file_name in N_imgs[round(len(N_imgs)/5):]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/N', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/train/N');

print("done") 
for file_name in V_imgs[0:round(len(V_imgs)/5)]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/V', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/test/V');
        
        
for file_name in V_imgs[round(len(V_imgs)/5):]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/V', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/train/V');

print("done")
for file_name in Q_imgs[0:round(len(Q_imgs)/5)]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/Q', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/test/Q');
        
        
for file_name in Q_imgs[round(len(Q_imgs)/5):]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/Q', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/train/Q');

print("done")
S_imgs = os.listdir('../input/ecg-images/MITBIH_img/S')

for file_name in S_imgs[0:round(len(S_imgs)/5)]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/S', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/test/S');
        
        
for file_name in S_imgs[round(len(S_imgs)/5):]:
    
    full_file_name = os.path.join('../input/ecg-images/MITBIH_img/S', file_name);
    
    if os.path.isfile(full_file_name):
        
        a = shutil.copy(full_file_name, '../working/train/S');

print("done")
folder = '/kaggle/working'
list_files(folder)
N_imgs       = len(os.listdir('../input/ecg-images/MITBIH_img/N'))
N_train_imgs = len(os.listdir('../working/train/N'))
N_test_imgs  = len(os.listdir('../working/test/N'))
print('number of N images at the original file: ', N_imgs)
print('total number of train and test picts:    ', N_train_imgs + N_test_imgs)
S_imgs       = len(os.listdir('../input/ecg-images/MITBIH_img/S'))
S_train_imgs = len(os.listdir('../working/train/S'))
S_test_imgs  = len(os.listdir('../working/test/S'))
print('number of S images at the original file: ', S_imgs)
print('total number of train and test picts:    ', S_train_imgs + S_test_imgs)
F_imgs       = len(os.listdir('../input/ecg-images/MITBIH_img/F'))
F_train_imgs = len(os.listdir('../working/train/F'))
F_test_imgs  = len(os.listdir('../working/test/F'))
print('number of F images at the original file: ', F_imgs)
print('total number of train and test picts:    ', F_train_imgs + F_test_imgs)
V_imgs       = len(os.listdir('../input/ecg-images/MITBIH_img/V'))
V_train_imgs = len(os.listdir('../working/train/V'))
V_test_imgs  = len(os.listdir('../working/test/V'))
print('number of V images at the original file: ', V_imgs)
print('total number of train and test picts:    ', V_train_imgs + V_test_imgs)
Q_imgs       = len(os.listdir('../input/ecg-images/MITBIH_img/Q'))
Q_train_imgs = len(os.listdir('../working/train/Q'))
Q_test_imgs  = len(os.listdir('../working/test/Q'))
print('number of Q images at the original file: ', Q_imgs)
print('total number of train and test picts:    ', Q_train_imgs + Q_test_imgs)
# Define default PATH

TRAIN_PATH        = '../working/train'

transform         = transforms.Compose(
                                       [transforms.Resize([120,120]),
                                        transforms.Grayscale(), 
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5))
                                       ])
  
train_data_set    = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)

batch_size=32

train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
TEST_PATH        = '../working/test'
  
test_data_set    = datasets.ImageFolder(root=TEST_PATH, transform=transform)

test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True)
# Run this to test your data loader

images, labels = next(iter(train_data_loader))
print(type(images))

print(images.size())

print("")
print("Batch Size:   ",images.size()[0])
print("Channel Size: ",images.size()[1])
print("Image Height: ",images.size()[2])
print("Image Width:  ",images.size()[3])
def imshow_tensor(image, ax=None, title=None, normalize=True):
    
    """Imshow for Tensor."""
    
    if ax is None:
        fig, ax = plt.subplots()
        
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5])
        std = np.array([0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
# show images

ncol = 8;

imshow_tensor(torchvision.utils.make_grid(images,nrow = ncol));
# print labels

classes = ('N', 'Q', 'F', 'S', 'V')

nrow = batch_size/ncol;

for row in range(int(nrow)):
    
    print(' '.join('%5s' % classes[labels[(row*ncol)+j]] for j in range(ncol))) 
# CNN Architect

class ConvNet_1(nn.Module):
    
    def __init__(self):
        
        super(ConvNet_1, self).__init__()

        self.layer_1  = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        
        self.relu1    = nn.ReLU(inplace=True)
        
        self.maxpool1 = MaxPool2d(kernel_size=2)
        

        self.layer_2  = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        
        self.relu2    = nn.ReLU(inplace=True)
        
        self.maxpool2 = MaxPool2d(kernel_size=2)
        
        self.drop_out = nn.Dropout()
        
        
        # out_channels = 4, number of classes = 5
        
        # image width = 120, image height = 120 after two maxpooling 120 -> 60 -> 30
        
        self.fc1 = nn.Linear(4 * 30 * 30, 5)
        
    # Defining the forward pass

    def forward(self, x):
        
        out = self.layer_1(x)
        
        out = self.relu1(out)
        
        out = self.maxpool1(out)
        
        
        out = self.layer_2(out)
        
        out = self.relu2(out)
        
        out = self.maxpool2(out)
        
        
        out = out.reshape(out.size(0), -1)
        
        out = self.drop_out(out)
        
        out = self.fc1(out)
        
        return out
    
# Define Model

model_1 = ConvNet_1()

print(model_1)
# Define Criterion

criterion = nn.CrossEntropyLoss()

# Define Optimizer

optimizer = optim.SGD(model_1.parameters(), lr=0.001, momentum=0.9)
# Whether to train on a gpu and Number of gpus

if cuda.is_available(): 
    
    print(f'{cuda.device_count()} number of gpus are detected and available.')
    
else:
        
    print(f'Train on gpu is not available')
        
        
# This part is working

# Train the model

if torch.cuda.is_available():
    
    MODEL = model_1.cuda()
    CRITERION = criterion.cuda()
    print(f'Model is started training on {cuda.device_count()} number of gpus.')
    print("Devise is cuda")
    
else:
    
    MODEL = model_1
    CRITERION = criterion
    print("Devise is cpu and model is started training.")

total_step = len(train_data_loader)
loss_list = []
acc_list = []

num_epochs = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_con_mat = torch.zeros([num_epochs, 5, 5], dtype=torch.int32, device=device)

for epoch in range(num_epochs):
    
    # define empty tensor 5*5 beginning of every epoch
    # tensor [row,column]
    con_mat = torch.zeros([5, 5], dtype=torch.int32, device=device)
    
    for i, data in enumerate(train_data_loader):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # optimization
        optimizer.zero_grad()
        
        # Forward to get output
        outputs = MODEL(inputs)
        # Calculate Loss
        loss = CRITERION(outputs, labels)
        # Backward propagation
        loss.backward()
        # Updating parameters
        optimizer.step()
        
        # Store loss
        loss_list.append(loss.item())
    
        # Calculate labels size
        total = labels.size(0)
        
        # Outputs.data has dimension batch size * 5
        # torch.max returns the max value of elements(_) and their indices(predicted) in the tensor array
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate total number of correct labels 
        correct = (predicted == labels).sum().item()
        
        # Store accuracy
        acc_list.append(correct / total)
        
        
        # Build Confusion Matrix
        for element in range(total):
            
            # con_mat[row,column]
            # con_mat[predictions, actual]
            
            con_mat[predicted[element].item()-1][labels[element].item()-1] += 1

        if (i + 1) % 300 == 0:                             # every 300 mini-batches...
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    print(con_mat)
            
    all_con_mat[epoch] = con_mat            
            
print('Finished Training')

plt.plot(loss_list);
plt.show();

plt.plot(acc_list);
plt.show();
# SUM FOR EACH COLUMN

# N Q S V F

print(torch.sum(con_mat, dim=0))
folder = '/kaggle/working'
list_files(folder)
# RECALL

# PRECISION

# F1-score = 2 × (precision × recall)/(precision + recall)

class_list = ['N', 'Q', 'S', 'V', 'F']

f1_score_list=[0,0,0,0,0]

precision_list=[0,0,0,0,0]

recall_list=[0,0,0,0,0]

delta = 0.0000000000001

for i in range(torch.sum(con_mat, dim=0).size(0)): 
    
    recall_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=0)[i].item()+delta)
    
    precision_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=1)[i].item()+delta)
    
    f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
    
    print('class: {:<2},total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.2f}%, F1-Score: {:.2f}%'
          
                  .format(class_list[i],
                          torch.sum(con_mat, dim=0)[i].item(),
                          con_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
# This part is working

if torch.cuda.is_available():
    
    MODEL = model_1.cuda()
    CRITERION = criterion.cuda()
    print("cuda")
    
else:
    
    MODEL = model_1
    CRITERION = criterion
    print("cpu")

# Train the model

total_step = len(train_data_loader)
loss_list = []
acc_list = []

num_epochs = 5

class_list = ['N', 'Q', 'S', 'V', 'F']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_con_mat = torch.zeros([num_epochs, 5, 5], dtype=torch.int32, device=device)

for epoch in range(num_epochs):
    
    f1_score_list=[0,0,0,0,0]

    precision_list=[0,0,0,0,0]

    recall_list=[0,0,0,0,0]
    
    delta = 0.0000000000001 
    
    # define empty tensor 5*5 beginning of every epoch
    # tensor [row,column]
    con_mat = torch.zeros([5, 5], dtype=torch.int32, device=device)
    
    for i, data in enumerate(train_data_loader):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # optimization
        optimizer.zero_grad()
        
        # Forward to get output
        outputs = MODEL(inputs)
        # Calculate Loss
        loss = CRITERION(outputs, labels)
        # Backward propagation
        loss.backward()
        # Updating parameters
        optimizer.step()
        
        # Store loss
        loss_list.append(loss.item())
    
        # Calculate labels size
        total = labels.size(0)
        
        # Outputs.data has dimension batch size * 5
        # torch.max returns the max value of elements(_) and their indices(predicted) in the tensor array
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate total number of correct labels 
        correct = (predicted == labels).sum().item()
        
        # Store accuracy
        acc_list.append(correct / total)
        
        for element in range(total):
            
            # con_mat[row,column]
            # con_mat[predictions, actual]
            con_mat[predicted[element].item()-1][labels[element].item()-1] += 1

        if (i + 1) % 300 == 0:                             # every 300 mini-batches...
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    print(con_mat)
            
    all_con_mat[epoch] = con_mat
    
    # Print Confusion Matrix
    
    for i in range(torch.sum(con_mat, dim=0).size(0)): 
    
        recall_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.2f}%, F1-Score: {:.2f}%'
          
                  .format(class_list[i],
                          torch.sum(con_mat, dim=0)[i].item(),
                          con_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
    
            
print('Finished Training')

plt.plot(loss_list);
plt.show();

plt.plot(acc_list);
plt.show();
all_con_mat
confusion_mat = torch.zeros([5, 5], dtype=torch.int32, device=device)

with torch.no_grad():
    
    for data in test_data_loader:
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = MODEL(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total = labels.size(0)
        
        # Calculate total number of correct labels 
        correct = (predicted == labels).sum().item()
        
        for element in range(total):
            
            # confusion_mat[row,column]
            # confusion_mat[predictions, actual]
            confusion_mat[predicted[element].item()-1][labels[element].item()-1] += 1

    print(confusion_mat)
class_list = ['N', 'Q', 'S', 'V', 'F']

f1_score_list=[0,0,0,0,0]

precision_list=[0,0,0,0,0]

recall_list=[0,0,0,0,0]
    
delta = 0.0000000000001 


for i in range(torch.sum(confusion_mat, dim=0).size(0)): 
    
        recall_list[i] = confusion_mat[i][i].item()/(torch.sum(confusion_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = confusion_mat[i][i].item()/(torch.sum(confusion_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.2f}%, F1-Score: {:.2f}%'
          
                  .format(class_list[i],
                          torch.sum(confusion_mat, dim=0)[i].item(),
                          confusion_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
    
# CNN Architect

class ConvNet_2(nn.Module):
    
    def __init__(self):
        
        super(ConvNet_2, self).__init__()

        self.layer_1  = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        
        self.relu1    = nn.ReLU(inplace=True)
        
        self.maxpool1 = MaxPool2d(kernel_size=2)
        

        self.layer_2  = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        self.relu2    = nn.ReLU(inplace=True)
        
        self.maxpool2 = MaxPool2d(kernel_size=2)
        
        self.drop_out = nn.Dropout()
        
        
        # out_channels = 8, number of classes = 5
        
        # image width = 120, image height = 120 after two maxpooling 120 -> 60 -> 30
        
        self.fc1 = nn.Linear(8 * 30 * 30, 5)
        
    # Defining the forward pass

    def forward(self, x):
        
        out = self.layer_1(x)
        
        out = self.relu1(out)
        
        out = self.maxpool1(out)
        
        
        out = self.layer_2(out)
        
        out = self.relu2(out)
        
        out = self.maxpool2(out)
        
        
        out = out.reshape(out.size(0), -1)
        
        out = self.drop_out(out)
        
        out = self.fc1(out)
        
        return out
    
# Define Model

model_2 = ConvNet_2()

print(model_2)
if torch.cuda.is_available():
    
    MODEL = model_2.cuda()
    CRITERION = criterion.cuda()
    print("cuda")
    
else:
    
    MODEL = model_2
    CRITERION = criterion
    print("cpu")

# Train the model

total_step = len(train_data_loader)
loss_list = []
acc_list = []

num_epochs = 5

class_list = ['N', 'Q', 'S', 'V', 'F']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_con_mat = torch.zeros([num_epochs, 5, 5], dtype=torch.int32, device=device)

for epoch in range(num_epochs):
    
    f1_score_list=[0,0,0,0,0]

    precision_list=[0,0,0,0,0]

    recall_list=[0,0,0,0,0]
    
    delta = 0.0000000000001 
    
    # define empty tensor 5*5 beginning of every epoch
    # tensor [row,column]
    con_mat = torch.zeros([5, 5], dtype=torch.int32, device=device)
    
    for i, data in enumerate(train_data_loader):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # optimization
        optimizer.zero_grad()
        
        # Forward to get output
        outputs = MODEL(inputs)
        # Calculate Loss
        loss = CRITERION(outputs, labels)
        # Backward propagation
        loss.backward()
        # Updating parameters
        optimizer.step()
        
        # Store loss
        loss_list.append(loss.item())
    
        # Calculate labels size
        total = labels.size(0)
        
        # Outputs.data has dimension batch size * 5
        # torch.max returns the max value of elements(_) and their indices(predicted) in the tensor array
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate total number of correct labels 
        correct = (predicted == labels).sum().item()
        
        # Store accuracy
        acc_list.append(correct / total)
        
        for element in range(total):
            
            # con_mat[row,column]
            # con_mat[predictions, actual]
            con_mat[predicted[element].item()-1][labels[element].item()-1] += 1

        if (i + 1) % 300 == 0:                             # every 300 mini-batches...
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    print(con_mat)
            
    all_con_mat[epoch] = con_mat
    
    # Print Confusion Matrix
    
    for i in range(torch.sum(con_mat, dim=0).size(0)): 
    
        recall_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.2f}%, F1-Score: {:.2f}%'
          
                  .format(class_list[i],
                          torch.sum(con_mat, dim=0)[i].item(),
                          con_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
    
            
print('Finished Training')

plt.plot(loss_list);
plt.show();

plt.plot(acc_list);
plt.show();
# CNN Architect

class ConvNet_3(nn.Module):
    
    def __init__(self):
        
        super(ConvNet_3, self).__init__()

        self.conv1    = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        
        self.relu1    = nn.ReLU(inplace=True)
        
        self.pool1    = MaxPool2d(kernel_size=2)
        

        self.conv2    = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        self.lrelu2   = nn.LeakyReLU(0.1)
        
        self.bn2      = nn.BatchNorm2d(8)
        
        self.pool2    = MaxPool2d(kernel_size=2)
        
        self.dropout2 = nn.Dropout(p=0.25)
        
        
        # out_channels = 8, number of classes = 5
        
        # image width = 120, image height = 120 after two maxpooling 120 -> 60 -> 30
        
        self.fc3      = nn.Linear(8 * 30 * 30, 100)
        
        self.relu3    = nn.ReLU(inplace=True)
        
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.fc4 = nn.Linear(100, 5)
        
    # Defining the forward pass

    def forward(self, x):
        
        out = self.conv1(x)
        
        out = self.relu1(out)
        
        out = self.pool1(out)
        
        
        
        out = self.conv2(out)
        
        out = self.lrelu2(out)
        
        out = self.bn2(out)
        
        out = self.pool2(out)
        
        out = self.dropout2(out)
        
        
        out = out.reshape(out.size(0), -1)
        
        
        out = self.fc3(out)
        
        out = self.relu3(out)
        
        out = self.dropout3(out)
        
        out = self.fc4(out)
        
        return out
    
# Define Model

model_3 = ConvNet_3()

print(model_3)
if torch.cuda.is_available():
    
    MODEL = model_3.cuda()
    CRITERION = criterion.cuda()
    print("cuda")
    
else:
    
    MODEL = model_3
    CRITERION = criterion
    print("cpu")

# Train the model

total_step = len(train_data_loader)
loss_list = []
acc_list = []

num_epochs = 5

class_list = ['N', 'Q', 'S', 'V', 'F']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_con_mat = torch.zeros([num_epochs, 5, 5], dtype=torch.int32, device=device)

for epoch in range(num_epochs):
    
    f1_score_list=[0,0,0,0,0]

    precision_list=[0,0,0,0,0]

    recall_list=[0,0,0,0,0]
    
    delta = 0.0000000000001 
    
    # define empty tensor 5*5 beginning of every epoch
    # tensor [row,column]
    con_mat = torch.zeros([5, 5], dtype=torch.int32, device=device)
    
    for i, data in enumerate(train_data_loader):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # optimization
        optimizer.zero_grad()
        
        # Forward to get output
        outputs = MODEL(inputs)
        # Calculate Loss
        loss = CRITERION(outputs, labels)
        # Backward propagation
        loss.backward()
        # Updating parameters
        optimizer.step()
        
        # Store loss
        loss_list.append(loss.item())
    
        # Calculate labels size
        total = labels.size(0)
        
        # Outputs.data has dimension batch size * 5
        # torch.max returns the max value of elements(_) and their indices(predicted) in the tensor array
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate total number of correct labels 
        correct = (predicted == labels).sum().item()
        
        # Store accuracy
        acc_list.append(correct / total)
        
        for element in range(total):
            
            # con_mat[row,column]
            # con_mat[predictions, actual]
            con_mat[predicted[element].item()-1][labels[element].item()-1] += 1

        if (i + 1) % 300 == 0:                             # every 300 mini-batches...
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    print(con_mat)
            
    all_con_mat[epoch] = con_mat
    
    # Print Confusion Matrix
    
    for i in range(torch.sum(con_mat, dim=0).size(0)): 
    
        recall_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.2f}%, F1-Score: {:.2f}%'
          
                  .format(class_list[i],
                          torch.sum(confusion_mat, dim=0)[i].item(),
                          confusion_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
    
            
print('Finished Training')

plt.plot(loss_list);
plt.show();

plt.plot(acc_list);
plt.show();
# CNN Architect

class ConvNet_4(nn.Module):
    
    def __init__(self):
        
        super(ConvNet_4, self).__init__()

        self.layer_1  = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        
        self.relu1    = nn.ReLU(inplace=True)
        
        self.maxpool1 = MaxPool2d(kernel_size=2)
        

        self.layer_2  = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding=2)
        
        self.relu2    = nn.ReLU(inplace=True)
        
        self.maxpool2 = MaxPool2d(kernel_size=2)
        
        self.drop_out = nn.Dropout()
        
        
        # out_channels = 4, number of classes = 5
        
        # image width = 120, image height = 120 after two maxpooling 120 -> 60 -> 30
        
        self.fc1 = nn.Linear(4 * 30 * 30, 5)
        
    # Defining the forward pass

    def forward(self, x):
        
        out = self.layer_1(x)
        
        out = self.relu1(out)
        
        out = self.maxpool1(out)
        
        
        out = self.layer_2(out)
        
        out = self.relu2(out)
        
        out = self.maxpool2(out)
        
        out = self.drop_out(out)
        
        
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        
        return out

# Define Model

model_4 = ConvNet_4()

print(model_4)
if torch.cuda.is_available():
    
    MODEL = model_4.cuda()
    CRITERION = criterion.cuda()
    print("cuda")
    
else:
    
    MODEL = model_4
    CRITERION = criterion
    print("cpu")

# Train the model

total_step = len(train_data_loader)
loss_list = []
acc_list = []

num_epochs = 5

class_list = ['N', 'Q', 'S', 'V', 'F']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_con_mat = torch.zeros([num_epochs, 5, 5], dtype=torch.int32, device=device)

for epoch in range(num_epochs):
    
    f1_score_list=[0,0,0,0,0]

    precision_list=[0,0,0,0,0]

    recall_list=[0,0,0,0,0]
    
    delta = 0.0000000000001 
    
    # define empty tensor 5*5 beginning of every epoch
    # tensor [row,column]
    con_mat = torch.zeros([5, 5], dtype=torch.int32, device=device)
    
    for i, data in enumerate(train_data_loader):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # optimization
        optimizer.zero_grad()
        
        # Forward to get output
        outputs = MODEL(inputs)
        # Calculate Loss
        loss = CRITERION(outputs, labels)
        # Backward propagation
        loss.backward()
        # Updating parameters
        optimizer.step()
        
        # Store loss
        loss_list.append(loss.item())
    
        # Calculate labels size
        total = labels.size(0)
        
        # Outputs.data has dimension batch size * 5
        # torch.max returns the max value of elements(_) and their indices(predicted) in the tensor array
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate total number of correct labels 
        correct = (predicted == labels).sum().item()
        
        # Store accuracy
        acc_list.append(correct / total)
        
        for element in range(total):
            
            # con_mat[row,column]
            # con_mat[predictions, actual]
            con_mat[predicted[element].item()-1][labels[element].item()-1] += 1

        if (i + 1) % 300 == 0:                             # every 300 mini-batches...
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    print(con_mat)
            
    all_con_mat[epoch] = con_mat
    
    # Print Confusion Matrix
    
    for i in range(torch.sum(con_mat, dim=0).size(0)): 
    
        recall_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.2f}%, F1-Score: {:.2f}%'
          
                  .format(class_list[i],
                          torch.sum(confusion_mat, dim=0)[i].item(),
                          confusion_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
    
            
print('Finished Training')

plt.plot(loss_list);
plt.show();

plt.plot(acc_list);
plt.show();
confusion_mat = torch.zeros([5, 5], dtype=torch.int32, device=device)

with torch.no_grad():
    
    for data in test_data_loader:
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = MODEL(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total = labels.size(0)
        
        # Calculate total number of correct labels 
        correct = (predicted == labels).sum().item()
        
        for element in range(total):
            
            # confusion_mat[row,column]
            # confusion_mat[predictions, actual]
            confusion_mat[predicted[element].item()-1][labels[element].item()-1] += 1

    print(confusion_mat)
class_list = ['N', 'Q', 'S', 'V', 'F']

f1_score_list=[0,0,0,0,0]

precision_list=[0,0,0,0,0]

recall_list=[0,0,0,0,0]
    
delta = 0.0000000000001 


for i in range(torch.sum(confusion_mat, dim=0).size(0)): 
    
        recall_list[i] = confusion_mat[i][i].item()/(torch.sum(confusion_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = confusion_mat[i][i].item()/(torch.sum(confusion_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.2f}%, F1-Score: {:.2f}%'
          
                  .format(class_list[i],
                          torch.sum(confusion_mat, dim=0)[i].item(),
                          confusion_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
