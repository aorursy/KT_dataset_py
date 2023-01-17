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

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
from math import sin, cos, pi
import cv2
from tqdm.notebook import tqdm
!unzip -u /kaggle/input/facial-keypoints-detection/training.zip
!unzip -u /kaggle/input/facial-keypoints-detection/test.zip
train_csv=pd.read_csv('training.csv')
test_csv=pd.read_csv('test.csv')
def get_info(data):
    images_=[]
    results_=[]
    for idx, sample in data.iterrows():
        image=sample['Image'].split(' ')
        image=np.array(image,dtype=int)
        image=image.reshape(96,96,1)
        images_.append(image)
    images_=np.array(images_)/255

    data=data.drop('Image',axis=1)
    for idx, sample in data.iterrows():
        results_.append(np.array(sample))
    results_=np.array(results_)

    return(images_,results_)
clean_train=train_csv.dropna()
unclean_train=train_csv.fillna(method='ffill')
(clean_image_train,clean_results_train)=get_info(clean_train)
(unclean_image_train,unclean_results_train)=get_info(unclean_train)
def plot_sample(image, keypoint, axis, title):
    image = image.reshape(96,96)
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)
fig, axis = plt.subplots()
plot_sample(unclean_image_train[150], unclean_results_train[150], axis, "Sample image & keypoints")
train_images=np.concatenate((clean_image_train,unclean_image_train))
train_results=np.concatenate((clean_results_train,unclean_results_train))
print(train_images.shape,train_results.shape)
print(train_results[5])
def swap_col(keypoints,n1,n2):
    keypoints[:,[n1,n2]]=keypoints[:,[n2,n1]]
    return keypoints
def flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2) 
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)]) 
    flipped_keypoints=np.array(flipped_keypoints)
    flipped_keypoints=swap_col(flipped_keypoints,0,2)
    flipped_keypoints=swap_col(flipped_keypoints,1,3)
    flipped_keypoints=swap_col(flipped_keypoints,4,8)
    flipped_keypoints=swap_col(flipped_keypoints,5,9)
    flipped_keypoints=swap_col(flipped_keypoints,6,10)
    flipped_keypoints=swap_col(flipped_keypoints,7,11)
    flipped_keypoints=swap_col(flipped_keypoints,12,16)
    flipped_keypoints=swap_col(flipped_keypoints,13,17)
    flipped_keypoints=swap_col(flipped_keypoints,14,18)
    flipped_keypoints=swap_col(flipped_keypoints,15,19)
    flipped_keypoints=swap_col(flipped_keypoints,22,24)
    flipped_keypoints=swap_col(flipped_keypoints,23,25)
    return flipped_images, flipped_keypoints
flip_clean_images,flip_clean_keypoints=flip(clean_image_train,clean_results_train)
train_images=np.concatenate((train_images,flip_clean_images))
train_keypoints=np.concatenate((train_results,flip_clean_keypoints))
print(train_images.shape,train_keypoints.shape)
def alter_brightness(images, keypoints):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*1.2, 0.0, 1.0)    # Increased brightness by a factor of 1.2 & clip any values outside the range of [-1,1]
    dec_brightness_images = np.clip(images*0.6, 0.0, 1.0)    # Decreased brightness by a factor of 0.6 & clip any values outside the range of [-1,1]
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return altered_brightness_images, np.concatenate((keypoints, keypoints))
(altered_brightness_images,altered_brightness_keypoints)=alter_brightness(clean_image_train,clean_results_train)
train_images=np.concatenate((train_images,altered_brightness_images))
train_keypoints=np.concatenate((train_keypoints,altered_brightness_keypoints))
print(train_images.shape,train_keypoints.shape)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_keypoints, test_size=0.2)
print(X_train.shape,X_valid.shape,y_train.shape,y_valid.shape)
X_train=np.transpose(X_train,[0,3,1,2])
X_valid=np.transpose(X_valid,[0,3,1,2])
print(X_train.shape,X_valid.shape)
X_train=torch.from_numpy(X_train).type(torch.FloatTensor)
X_valid=torch.from_numpy(X_valid).type(torch.FloatTensor)
y_train=torch.from_numpy(y_train).type(torch.FloatTensor)
y_valid=torch.from_numpy(y_valid).type(torch.FloatTensor)
batch_size=8
n_iter=100000
n_epoch=int(n_iter/(y_train.shape[0]/batch_size))

train=torch.utils.data.TensorDataset(X_train,y_train)
valid=torch.utils.data.TensorDataset(X_valid,y_valid)
train_loader=DataLoader(train,batch_size=batch_size,shuffle=True)
valid_loader=DataLoader(valid,batch_size=batch_size,shuffle=False)
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.pool=nn.MaxPool2d(2)
        
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.BN1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.BN2=nn.BatchNorm2d(32)
        self.drop1=nn.Dropout(0.4)
        
        self.conv3=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.BN3=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.BN4=nn.BatchNorm2d(64)
        self.drop2=nn.Dropout(0.4)
        
        self.conv5=nn.Conv2d(64,96,kernel_size=3,padding=1)
        self.BN5=nn.BatchNorm2d(96)
        self.conv6=nn.Conv2d(96,96,kernel_size=3,padding=1)
        self.BN6=nn.BatchNorm2d(96)
        self.drop3=nn.Dropout(0.4)
        
        self.conv7=nn.Conv2d(96,128,kernel_size=3,padding=1)
        self.BN7=nn.BatchNorm2d(128)
        self.conv8=nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.BN8=nn.BatchNorm2d(128)
        self.drop4=nn.Dropout(0.4)
        
        self.conv9=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.BN9=nn.BatchNorm2d(256)
        self.conv10=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.BN10=nn.BatchNorm2d(256)
        self.drop5=nn.Dropout(0.4)
        
        self.conv11=nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.BN11=nn.BatchNorm2d(512)
        self.conv12=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.BN12=nn.BatchNorm2d(512)
        self.drop6=nn.Dropout(0.4)
        
        self.LN1=nn.Linear(512*3*3,512)
        self.BN13=nn.BatchNorm1d(512)
        self.drop7=nn.Dropout(0.4)
        self.LN2=nn.Linear(512,256)
        self.LN3=nn.Linear(256,30)
        
    def forward(self,x):
        
        x=F.relu(self.conv1(x))
        x=self.BN1(x)
        x=F.relu(self.conv2(x))
        x=self.BN2(x)
        x=self.pool(x)
        x=self.drop1(x)
        
        x=F.relu(self.conv3(x))
        x=self.BN3(x)
        x=F.relu(self.conv4(x))
        x=self.BN4(x)
        x=self.pool(x)
        x=self.drop2(x)
        
        x=F.relu(self.conv5(x))
        x=self.BN5(x)
        x=F.relu(self.conv6(x))
        x=self.BN6(x)
        x=self.pool(x)
        x=self.drop3(x)
        
        x=F.relu(self.conv7(x))
        x=self.BN7(x)
        x=F.relu(self.conv8(x))
        x=self.BN8(x)
        x=self.pool(x)
        x=self.drop4(x)
        
        x=F.relu(self.conv9(x))
        x=self.BN9(x)
        x=F.relu(self.conv10(x))
        x=self.BN10(x)
        x=self.pool(x)
        x=self.drop5(x)
        
        x=F.relu(self.conv11(x))
        x=self.BN11(x)
        x=F.relu(self.conv12(x))
        x=self.BN12(x)
        x=self.drop6(x)
        
        x=x.view(-1,512*3*3)
        x=F.relu(self.LN1(x))
        x=self.BN13(x)
        x=self.drop7(x)
        x=F.relu(self.LN2(x))
        x=self.LN3(x)
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)

model=Net()
model.apply(weight_init)
model.to(device)
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
his_train=[]
his_val=[]
num_iter=0

for epoch in range(n_epoch):  # loop over the dataset multiple times
    train_cnt=0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        train_cnt+=1
        inputs, labels = data
        #print(inputs.shape,labels.shape)
        inputs, labels = inputs.to(device),labels.to(device)
        num_iter+=1
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        print(type(outputs),type(labels))
        loss = torch.sqrt(criterion(outputs, labels))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
#if num_iter %100== 99:    # print every 100 mini-batches
    print('train loss[%6d]: %.6f' %(num_iter,running_loss / train_cnt))
    
    cnt=0
    valid_loss = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.sqrt(criterion(outputs, labels))
        valid_loss += loss.item()
        cnt+=1
    print('valid loss: %.6f'%(valid_loss/cnt))
    his_train.append((running_loss / train_cnt))
    his_val.append((valid_loss/cnt))
print('Finished Training')
