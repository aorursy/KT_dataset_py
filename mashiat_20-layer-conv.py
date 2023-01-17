! git clone https://github.com/dwgoon/jpegio
# Once downloaded install the package
!pip install jpegio/.
import jpegio as jio
import jpegio as jpio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,datasets

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os 
import cv2
import numpy as np
from tqdm.notebook import tqdm
import sys

REBUILD_DATA= True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
# Image preprocessing modules
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize(512, 512),
    transforms.ToTensor()])
# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
train_imageids = pd.Series(os.listdir(BASE_PATH + '/Cover')).sort_values(ascending=True).reset_index(drop=True)
test_imageids = pd.Series(os.listdir(BASE_PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)
sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')
test_transform=transforms.Compose([
    transforms.Resize(512, 512),
    transforms.ToTensor()])
print(test_imageids.head(3))
train_imageids.head()
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(BASE_PATH, pre, file))
train_filenames = np.array(os.listdir("/kaggle/input/alaska2-image-steganalysis/Cover/"))
print(len(train_filenames))
train_filenames
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
np.random.seed(0)
positives = train_filenames.copy()
negatives = train_filenames.copy()
np.random.shuffle(positives)
np.random.shuffle(negatives)

jmipod = append_path('JMiPOD')(positives[:10000])
juniward = append_path('JUNIWARD')(positives[10000:20000])
uerd = append_path('UERD')(positives[20000:30000])

pos_paths = np.concatenate([jmipod, juniward, uerd])
test_paths = append_path('Test')(sub.Id.values)
neg_paths = append_path('Cover')(negatives[:30000])
train_paths = np.concatenate([pos_paths, neg_paths])
train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
print(train_paths)
from sklearn.model_selection import train_test_split
#https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020)
print(len(train_paths))
print(len(valid_paths))
l=np.array([train_paths,train_labels])
traindataset = pd.DataFrame({ 'images': list(train_paths), 'label': train_labels},columns=['images','label'])
traindataset
val_l=np.array([valid_paths,valid_labels])
validdataset=dataset = pd.DataFrame({ 'images': list(valid_paths), 'label': valid_labels},columns=['images','label'])
validdataset
from PIL import Image, ImageFile
from tqdm import tqdm_notebook as tqdm
import scipy                        # for cosine similarity
from scipy import fftpack
# add image augmen tation
class train_images(torch.utils.data.Dataset):

    def __init__(self, csv_file):

        self.data = csv_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(idx)
        img_name =  self.data.loc[idx][0]
        jpegStruct = jpio.read(img_name)
        image=np.zeros([512, 512, 3])
        for j in range(3):
            image[:,:,j]=jpegStruct.coef_arrays[j]
        #image = Image.open(img_name)
        label = self.data.loc[idx][1] #torch.tensor(self.data.loc[idx, 'label'])
       

# ## https://pytorch.org/docs/stable/torchvision/transforms.html
# transforms.Compose([
# transforms.CenterCrop(10),
# transforms.ToTensor(),
# ])
        
#         return {'image': transforms.ToTensor()(image), # ORIG
        return {'image': image,
            'label': label
            }

train_dataset = train_images(traindataset)
valid_dataset = train_images(validdataset)
print(type(train_dataset))
len(train_dataset)
batch_size = 32
epochs = 2
learning_rate = 0.001
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, num_workers=4)
def conv3x3(in_channels, out_channels, stride=1, padding = 0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding = padding, bias=False)
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.expansion=1
        self.conv1 = conv3x3(in_channels, in_channels, stride=1, padding = 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        
        self.pad = 1
        if stride!=1:
            self.expansion=2
            
        self.conv2 = conv3x3(in_channels, out_channels ,stride, padding = self.pad)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x,):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.expansion==2:
            residual = self.conv2(residual)
            residual = self.bn2(residual)
            
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, dct_input_channel,  num_classes=2):
        super(ResNet, self).__init__()
        
        self.in_channels = 24
        self.conv1 = conv3x3(dct_input_channel, int(self.in_channels/2), padding = 1)
        self.bn1 = nn.BatchNorm2d(int(self.in_channels/2))
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(int(self.in_channels/2), self.in_channels, stride =2)
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        
        self.conv= conv3x3(dct_input_channel, self.in_channels, stride = 2)
        
        self.layer1 = self.make_layer(block, self.in_channels*2) #in_channels = 24 
        self.layer2 = self.make_layer(block, self.in_channels*2)
        self.layer3 = self.make_layer(block, self.in_channels*2)
        self.layer4 = self.make_layer(block, self.in_channels*2)
        
        self.avg_pool = nn.AvgPool2d(16)
        self.fc = nn.Linear(self.in_channels , num_classes)
        
    def make_layer(self, block, out_channels):
        layers = []
        layers.append(block(self.in_channels, self.in_channels , stride =1))
        layers.append(block(self.in_channels, out_channels , stride =2))
        self.in_channels=out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        residual= self.relu(self.bn2(self.conv(x)))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out+=residual
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
model = ResNet(IdentityBlock, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)
epoch_loss = 0
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(epochs):

    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    counter = 0
    for bi, d in enumerate(tk0):
            inputs = d["image"]
            labels = d["label"].view(-1, 1)
    
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            #labels = labels.squeeze(1)
        
            inputs = inputs.view(-1,3,512,512)
            #print(inputs.shape)
            
            # Forward pass
            outputs = model(inputs)
            loss  = criterion(outputs, torch.max(labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss =epoch_loss + loss.item()
            
    epoch_loss = epoch_loss / (len(train_loader)/batch_size)
    print ("Epoch [{}/{}] Loss: {:.4f}".format(epoch+1, epochs, epoch_loss))

    # Decay learning rate
    curr_lr /= 3
    update_lr(optimizer, curr_lr)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    tk0 = tqdm(valid_loader, total=int(len(valid_loader)))
    counter = 0
    for bi, d in enumerate(tk0):
            inputs = d["image"]
            labels = d["label"].view(-1, 1)
    
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            inputs = inputs.view(-1,3,512,512)
   
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
    
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i,val in enumerate(predicted):
                print(val)
    print("correct",correct)
    print("total", total)
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
torch.save(model.state_dict(), 'resnet.ckpt')
