# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision.utils import make_grid

from torchvision import datasets,models,transforms

from PIL import Image

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
use_gpu = torch.cuda.is_available()

use_gpu
root = "../input/chest-xray-pneumonia/chest_xray/"
img_names = []

for folder,subfolders,filenames in os.walk(root):

    for img in filenames:

        img_names.append(folder+'/'+img)

print('images length:', len(img_names))
img_sizes =[]

rejected = []

for item in img_names:

    try:

        with Image.open(item) as img:

            img_sizes.append(img.size)

    except:

        rejected.append(item)

print(f'images:{len(img_sizes)}')

print(f'rejected:{len(rejected)}')
df = pd.DataFrame(img_sizes)

df.describe()
img1 = Image.open("../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0007-0001.jpeg")

display(img1)

img1.size
train_transform = transforms.Compose([

    transforms.Resize(224),

    transforms.CenterCrop(224),

    transforms.ToTensor()

])

test_transform = transforms.Compose([

    transforms.Resize(224),

    transforms.CenterCrop(224),

    transforms.ToTensor()

])
train_data = datasets.ImageFolder(os.path.join(root,'train'),transform  = train_transform)

test_data = datasets.ImageFolder(os.path.join(root,'test')  , transform = test_transform)



train_loader = DataLoader(train_data,batch_size=10,shuffle = True,num_workers =4)

test_loader = DataLoader(test_data,batch_size =10,shuffle = False,num_workers =4)



class_names = train_data.classes

print(len(train_data))

print(len(test_data))

print(class_names)
im
for images,labels in train_loader:

    break

print('labels:',*labels.numpy())

print('images:',*np.array([class_names[i] for i in labels]))

im= make_grid(images,nrow = 10)

plt.figure(figsize =(10,8))

plt.imshow(np.transpose(im.numpy(),(1,2,0)))



print("images.shape:",images.shape)

print("labels.shape:",labels.shape)

print("len(labels)",len(labels))
class ConvolutionalNNetworks(nn.Module):

    def __init__(self):

        super(ConvolutionalNNetworks,self).__init__()

        self.cnn_layers= nn.Sequential(

            nn.Conv2d(3,32,kernel_size=3,stride= 1,padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace = True),

            nn.Conv2d(32,32,kernel_size =3,stride = 1,padding = 1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace = True),

            nn.MaxPool2d(kernel_size =2,stride = 2))

        

        self.linear_layers = nn.Sequential(

            nn.Dropout(p = 0.5),

            nn.Linear(32*112*112,128),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace = True),

            nn.Dropout(p = 0.5),

            nn.Linear(128,2))

    def forward(self,X):

        X = self.cnn_layers(X)

        X = X.view(-1,32*112*112)

        X = self.linear_layers(X)

        return F.log_softmax(X,dim=1)
CNN_model = ConvolutionalNNetworks()



optimizer = torch.optim.Adam(CNN_model.parameters(),lr = 0.01)

criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():

    CNN_model = CNN_model.cuda()

    criterion = criterion.cuda()

print(CNN_model)
########################

## Training the model ##

########################

def train(epoch):

    CNN_model.train()

    trn_loss =0

    trn_corr =0

    total = 0

    for batch,(data,target) in enumerate(train_loader):

        data,target = Variable(data),Variable(target)

        if torch.cuda.is_available():

            data = data.cuda()

            target = target.cuda()

        

        y_pred = CNN_model(data)

        

        predicted = torch.max(y_pred.data,1)[1]

        trn_corr +=(predicted == target).sum()

        total += len(data)

        

        loss = criterion(y_pred,target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (batch+1)%100 ==0:

            print(f'train epoch: {epoch} loss: {loss.item()} accuracy:{trn_corr.item()/(len(data)*(batch+1))}')

    train_losses.append(loss)

    train_correct.append(trn_corr)

    

        

        

        

        

        
########################

## Testing the model ##

########################

def test(test_loader):

    CNN_model.eval()

    tst_loss =0

    tst_corr =0

    

    with torch.no_grad():

        for data,target in test_loader:

            if torch.cuda.is_available():

                data = data.cuda()

                target = target.cuda()

            

            y_eval = CNN_model(data)

            predicted = torch.max(y_eval.data,1)[1]

            tst_corr+=(predicted == target).sum()

        

        loss = criterion(y_eval,target)

        test_losses.append(loss)

        test_correct.append(tst_corr)

    

    

    
n_epochs = 1

train_losses= []

test_losses = []

train_correct= []

test_correct = []

for epoch in range(n_epochs):

    train(epoch)

    test(test_loader)
pwd
# Saving trained model as .pt

torch.save(CNN_model.state_dict(),'/kaggle/working/My_CNN_model.pt')
# Loading the trained model

model_CNN = ConvolutionalNNetworks()

model_CNN.load_state_dict(torch.load('/kaggle/working/My_CNN_model.pt'))

model_CNN.eval()
def plot_graph(epochs):

    fig = plt.figure(figsize=(20,4))

    ax = fig.add_subplot(1,2,1)

    plt.title("Train - Validation Loss")

    plt.plot(list(np.arange(epochs) + 1) , train_losses, label='train')

    plt.plot(list(np.arange(epochs) + 1), test_losses, label='validation')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('loss', fontsize=12)

    plt.legend(loc='best')

    

    ax = fig.add_subplot(1, 2, 2)

    plt.title("Train - Validation Accuracy")

    plt.plot(list(np.arange(epochs) + 1) , train_correct, label='train')

    plt.plot(list(np.arange(epochs) + 1), test_correct, label='validation')

    plt.xlabel('num_epochs', fontsize=12)

    plt.ylabel('accuracy', fontsize=12)

    plt.legend(loc='best')
plot_graph(n_epochs)
with torch.no_grad():

    out = CNN_model(train_data[2990][0].view(1,3,224,224).cuda()).argmax()

print('predicted:{}'.format(out.item()))

print('label:',train_data[2990][1])
with torch.no_grad():

    out = CNN_model(train_data[2990][0].view(1,3,224,224)).argmax()

print('predicted:{}'.format(out.item()))

print('label:',train_data[2990][1])