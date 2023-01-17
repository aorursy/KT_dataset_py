# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('/kaggle/input'))

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

      #  print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

from torchvision import datasets,transforms

tf=transforms.Compose([transforms.ToTensor()])

data = datasets.ImageFolder('/kaggle/input/four-shapes/shapes',transform = tf)
data.classes
type(data)
final_data= torch.utils.data.DataLoader(data,batch_size=10,shuffle=True)

#train_size=(int)(len(final_data)*0.7)

#train,test=torch.utils.data.random_split(final_data,[train_size,len(final_data)-train_size])

#train_data=torch.utils.data.DataLoader(train,batch_size=10)

#test=torch.utils.data.DataLoader(test)
len(final_data)
train_data.dataset

class CNN(nn.Module):

    def __init__(self):

        super(CNN,self).__init__()

        #in_channels, out_channels, kernel_size, stride=1, padding=0

        #size of output=(input_size+2Xpadding-filter_size)/stride

        placeholder=3

        self.conv1=nn.Conv2d(placeholder,8,7,stride=1)

        self.pool1=nn.MaxPool2d(2,2)

        self.relu1=nn.ReLU()

        self.conv2=nn.Conv2d(8,12,5)

        self.pool2=nn.MaxPool2d(2,2)

        self.relu2=nn.ReLU()

        self.fc1=nn.Linear(12*46*46,100)

        self.dropout=nn.Dropout(0.4)                           

        self.fc2=nn.Linear(100,4)

    

    def forward(self,x):

        #print('Original Size: ',x.size())

        x=self.conv1(x)

        #print('After 1st Convolutional Layer',x.size())

        x=self.pool1(x)

        #print('After Max Pooling',x.size())

        x=self.relu1(x)

        x=self.conv2(x)

        #print('After 2nd Convolutional Layer',x.size())

        x=self.pool2(x)

        #print('After 2nd Max Pooling',x.size())

        x=self.relu2(x)

       

        x=x.view(-1,12*46*46)

        #print('The Fully Connected Layer',x.size())

        x=self.dropout(x)

        #print('After Droput',x.size())

        x=self.relu1(self.fc1(x))

       

        x=self.relu1(self.fc2(x))

        return x

        

CNN()
import torch.optim as optim

cnn=CNN()



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(cnn.parameters(), lr=0.0001)

epochs=5

#ll=enumerate(final_data,0)

#next(ll)

loss_val=[]

loss_show=[]

for epoch in range(epochs):

    print("\nepoch: "+str(epoch),end=': ')

    for idx,datas in enumerate(final_data,0):

        ip,labels=datas

        optimizer.zero_grad()

        op=cnn.forward(ip)

        loss=criterion(op,labels)

        loss.backward()

        optimizer.step()

        loss_val.append(loss.item())

        if idx%500==0:

            loss_show.append(sum(loss_val)/len(loss_val))

            print(idx,end='|')

            loss_val=[]
ip.size()
#cnn=CNN()

tt=cnn.forward(x=ip)
#len(final_data)

labels
import matplotlib.pyplot as plt

plt.plot(range(len(loss_show)),loss_show)

plt.title('Training Loss')

plt.ylabel('Cross-Entropy Loss')
torch.save(cnn,'model_params')
cnn.forward(ip)
image=ip[4,:,:,:]


#image=image.numpy()

image.shape

act=np.zeros((200,200,3))

act.shape

act[:,:,0]=image[0,:,:]

act[:,:,1]=image[1,:,:]

act[:,:,2]=image[2,:,:]

#image=image*255

plt.imshow(act)

#cv2.waitKey()

print("Actual: ",data.classes[labels[4]])

print("Predicted: ",data.classes[torch.max(cnn.forward(ip),1)[1][4]])
image=ip[3,:,:,:]

image.shape

act=np.zeros((200,200,3))

act.shape

act[:,:,0]=image[0,:,:]

act[:,:,1]=image[1,:,:]

act[:,:,2]=image[2,:,:]

#image=image*255

plt.imshow(act)

#cv2.waitKey()

print("Actual: ",data.classes[labels[3]])

print("Predicted: ",data.classes[torch.max(cnn.forward(ip),1)[1][3]])