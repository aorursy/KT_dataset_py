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
import torchvision
import numpy
import matplotlib.pyplot as plt
import pandas as pd
mnist_train=pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
mnist_train.head()
random_sample=mnist_train.sample(8)
image_features=random_sample.drop('label',axis=1)
image_batch=(torch.tensor(image_features.values/255)).reshape(-1,28,28)
image_batch.shape
grid=torchvision.utils.make_grid(image_batch.unsqueeze(1),nrow=8)
print(grid.shape)
plt.figure(figsize=(12,12))
plt.imshow(grid.numpy().transpose(1,2,0))
plt.axis('off')
train_features=mnist_train.drop('label',axis=1)
train_label=mnist_train['label']

from torch.utils.data import Dataset,DataLoader
# torch.from_numpy(image_features.values[0])
#custom datasets class must implement __getitem__ and __len__ methods
class MnistDataset(Dataset):
    def __init__(self,path,transform=None):
        #loading dataframe as xy
        xy=pd.read_csv(path)
        #x is for the image matrix
        self.x=xy.iloc[:,1:].values
        #y is for the label
        self.y=xy['label']
        #n_samples gives number of images
        self.n_samples=len(xy)
        self.transform=transform
        
    def __getitem__(self,index):
        sample=self.x[index],np.array(self.y[index])
        if self.transform is not None:
            sample=self.transform(sample)
        return sample
    def __len__(self):
        return self.n_samples

#custom transformation class  must implement __call__
class ToTensor:
    def __call__(self,sample):
        inputs,target=sample
        #from_numpy converts numpy arrays to tensors
        return torch.from_numpy(inputs),torch.from_numpy(target)
        
train_path='/kaggle/input/mnist-in-csv/mnist_train.csv'
test_path='/kaggle/input/mnist-in-csv/mnist_test.csv'
train_set=MnistDataset(train_path,transform=ToTensor())
# test_set=MnistDataset(test_path)
first_data=train_set[0]
plt.imshow(first_data[0].numpy().reshape(28,28))
from torchvision import transforms

train_loader=DataLoader(train_set,batch_size=100,shuffle=True)
# test_loader=DataLoader(test_set,batch_size=100,shuffle=True)
plt.imshow(iter(train_loader).next()[0][0].reshape(28,28))
iter(train_loader).next()[0][0].dtype
#it is 1 because of single channel of colour i.e Black&White or Grayscale image
input_size=1
#first convolution converts 1 channel to 16 channels in feature maps
hid1_size=16
#similary to 32 channels
hid2_size=32

k_conv_size=5# kernel or filter size
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1=torch.nn.Sequential(
        torch.nn.Conv2d(input_size,hid1_size,k_conv_size),
        torch.nn.BatchNorm2d(hid1_size),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2)
            
        )
        
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(hid1_size,hid2_size,k_conv_size),
            torch.nn.BatchNorm2d(hid2_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.fc=torch.nn.Linear(512,10)
        
        
    def forward(self,x):
        x=self.layer1(x)
        
        x=self.layer2(x)
        #Changing the image into one dimensional tensor for feeding the fully connected layers
        x=x.reshape(x.shape[0],-1)
       
        x=self.fc(x)
        return x
model=Net()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#It is to check model accepts the input or not
# x=torch.randn(100,1,28,28)
# model(x)
# model
lr=1e-3
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)

epochs=3
loss_values=[]
targets=np.array([])
preds=np.array([])
for epoch in range(epochs):
    for i,(img,target) in enumerate(train_loader):
        img=img.reshape(100,1,28,28).float().to(device)
        optimizer.zero_grad()
        output=model(img)
        pred=torch.argmax(output,axis=1)
#         print(target,pred)
        targets=np.hstack([targets,target.cpu().numpy()])
        preds=np.hstack([preds,pred.cpu().numpy()])
        loss=loss_fn(output,target.to(device))
        
        loss.backward()
        optimizer.step()
        if i % 100==0:
            print(loss)

from sklearn.metrics import accuracy_score,recall_score
accuracy_score(targets,preds)
recall_score(targets,preds,average='macro')
