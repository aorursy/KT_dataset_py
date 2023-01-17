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
import torchvision
from torchvision import transforms, datasets, models

import torch
import matplotlib.pyplot as plt
import os
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d
from torch.nn import Module, Softmax, BatchNorm2d, Dropout

import gc
os.listdir('../input/ptbdb-ecg/PTB/train')
ECG_list = os.listdir('../input/ptbdb-ecg/PTB/train')

n_classes = len(ECG_list)

print(f'There are {n_classes} different classes.')
ECG_list
N_imgs = os.listdir('../input/ptbdb-ecg/PTB/train/N')
print('# of Normal beats: ',len(N_imgs))
M_imgs = os.listdir('../input/ptbdb-ecg/PTB/train/M')
print('number of mycardial infarction beats: ',len(M_imgs))
def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
import matplotlib.image as mpimg
image = mpimg.imread(os.path.join('../input/ptbdb-ecg/PTB/train/M', M_imgs[0]))

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
image_shows(folder = '../input/ptbdb-ecg/PTB/train/N', number_of_images = 6)
import shutil
from os import walk
len(os.listdir('../input/ptbdb-ecg/PTB/train/M')), len(os.listdir('../input/ptbdb-ecg/PTB/test/M'))

len(os.listdir('../input/ptbdb-ecg/PTB/train/N')), len(os.listdir('../input/ptbdb-ecg/PTB/test/N'))
# no. of files

def list_files(startpath):
    
    for root, dirs, files in os.walk(startpath):
        
        level = root.replace(startpath, '').count(os.sep)
        
        indent = ' ' * 4 * (level)
        
        print('{}{}'.format(indent, os.path.basename(root)), '-', len(os.listdir(root)))
        
folder = '../input/ptbdb-ecg/PTB/'
list_files(folder)
def imshow(image):
    """Display image"""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
image = mpimg.imread(os.path.join('../input/ptbdb-ecg/PTB/train/M', M_imgs[0]))

imshow(image)
print(image.shape)
print(type(image))
# Define default PATH

TRAIN_PATH        = '../input/ptbdb-ecg/PTB/train'

transform         = transforms.Compose(
                                       [transforms.Resize([64,64]),
                                        transforms.Grayscale(num_output_channels=3), 
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5))
                                       ])
  
train_data_set    = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)

batch_size=32

train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
TEST_PATH        = '../input/ptbdb-ecg/PTB/test'
  
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
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
device
import torchvision
model = models.wide_resnet50_2()
n_fltrs=model.fc.in_features
model.fc=nn.Linear(n_fltrs,2)
model.to(device)
# Define Criterion

criterion = nn.CrossEntropyLoss()

# Define Optimizer

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Whether to train on a gpu and Number of gpus

if cuda.is_available(): 
    
    print(f'{cuda.device_count()} number of gpus are detected and available.')
    
else:
        
    print(f'Train on gpu is not available')
%%time


# This part is working

if torch.cuda.is_available():
    
    MODEL = model.cuda()
    CRITERION = criterion.cuda()
    print("cuda")
    
else:
    
    MODEL = model
    CRITERION = criterion
    print("cpu")

# Train the model

total_step = len(train_data_loader)
loss_list = []
acc_list = []

num_epochs = 5

class_list = ['N', 'M']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_con_mat = torch.zeros([num_epochs, 2, 2], dtype=torch.int32, device=device)

for epoch in range(num_epochs):
    
    f1_score_list=[0,0]

    precision_list=[0,0]

    recall_list=[0,0]
    
    delta = 0.0000000000001 
    
    # define empty tensor 5*5 beginning of every epoch
    # tensor [row,column]
    con_mat = torch.zeros([2, 2], dtype=torch.int32, device=device)
    
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

        if (i + 1) % 70 == 0:                             # every 300 mini-batches...
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%'
                  
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    print(con_mat)
            
    all_con_mat[epoch] = con_mat
    
    # Print Confusion Matrix
    
    for i in range(torch.sum(con_mat, dim=0).size(0)): 
    
        recall_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = con_mat[i][i].item()/(torch.sum(con_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.2f}%, Precision: {:.4f}%, F1-Score: {:.4f}%'
          
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
import pandas as pd
df4 = pd.DataFrame(loss_list)
df4.to_csv("train_4.csv", index=False)
import pandas as pd
df2 = pd.DataFrame(acc_list)
df2.to_csv("train_acc.csv", index=False)
with plt.style.context("seaborn-poster"):
    fig, ax = plt.subplots(figsize=(12, 5));
    plt.plot(df2,color='red',  marker='o',markerfacecolor='r', label="Training Acc");
    #plt.plot(df4,  marker='*', label="Validation Acc")
    #plt.xticks(np.arange(0, 201, 20),fontweight='bold')
    plt.yticks(fontweight='bold');
    plt.ylabel('Accuracy', fontsize=18, fontweight='bold');
    plt.xlabel('epochs', fontsize=18, fontweight='bold')
    plt.title("Accuracy history using CNN+LSTM", fontweight='bold');
    plt.legend();
    #plt.savefig(f"loss_history.svg",format="svg",bbox_inches='tight', pad_inches=0.2)
    #plt.savefig(f"loss_history.png", format="png",bbox_inches='tight', pad_inches=0.2) 
    plt.show()


"""with plt.style.context("seaborn-poster"):
    plt.plot(df4, color='red', linewidth=5, marker='o',
    markerfacecolor='k', markersize=12)
    plt.show()""";
with plt.style.context("seaborn-poster"):
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.plot(df4,  color='red', marker='o',  label="Training Loss")
    #plt.plot(df4["val_loss"],  marker='o', label="Validation Loss")
    #plt.xticks(np.arange(0, 201, 20),fontweight='bold')
    #plt.yticks(fontweight='bold')
    plt.ylabel('loss', fontsize=18, fontweight='bold')
    plt.xlabel('epochs', fontsize=18, fontweight='bold')
    plt.title("Loss history using SMOTE+Tomek+CNN+LSTM", fontweight='bold')
    plt.legend()
    #plt.savefig(f"loss_history.svg",format="svg",bbox_inches='tight', pad_inches=0.2)
    #plt.savefig(f"loss_history.png", format="png",bbox_inches='tight', pad_inches=0.2) 
    plt.show()
%%time


confusion_mat = torch.zeros([2, 2], dtype=torch.int32, device=device)

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
class_list = ['N', 'M']

f1_score_list=[0,0]

precision_list=[0,0]

recall_list=[0,0]
    
delta = 0.0000000000001 


for i in range(torch.sum(confusion_mat, dim=0).size(0)): 
    
        recall_list[i] = confusion_mat[i][i].item()/(torch.sum(confusion_mat, dim=0)[i].item()+delta)
    
        precision_list[i] = confusion_mat[i][i].item()/(torch.sum(confusion_mat, dim=1)[i].item()+delta)
    
        f1_score_list[i] = 2 * precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i]+delta)
        
    
        print('class name: {}, total number of class: {:>5}, Correctly predicted: {:>5}, Recall: {:.4f}%, Precision: {:.4f}%, F1-Score: {:.4f}%'
          
                  .format(class_list[i],
                          torch.sum(confusion_mat, dim=0)[i].item(),
                          confusion_mat[i][i].item(), 
                          recall_list[i],
                          precision_list[i],
                          f1_score_list[i]
                         ))
!pip install git+https://github.com/qubvel/segmentation_models.pytorch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
from segmentation_models_pytorch.unet import Unet
model = Unet(encoder_name="efficientnet-b0", classes=2, aux_params={"classes": 2})
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()
