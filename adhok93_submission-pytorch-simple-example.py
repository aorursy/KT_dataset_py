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

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



# import torchvision module to handle image manipulation

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

# calculate train time, writing train data to files etc.

import time

import pandas as pd

import json

import numpy as np
class Dataloader(Dataset):

    

    def __init__(self,image,label,is_train=True):

        

        self.img = image

        self.label = label

        self.is_train = is_train

        

    def __len__(self):

        return len(self.img)

    

    def __getitem__(self,idx):

        

        '''

        Reshape: 1D) 784 -> 2D) 28x28

        '''

        image1 = self.img[idx].reshape(-1,28,28)

        if self.is_train:

            label1 = np.zeros(10, dtype='float32')

            label1[self.label[idx]] =1

            #label1 = self.label

            return image1,label1

        else:

            return image1

        



train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

train_data = train_data.sample(frac=1,random_state = 42)

test_data = train_data[55000:]





train_data = train_data[0:54999]



train_label = train_data['label'].values

print(train_label)

train_matrix = train_data.drop('label',axis=1).values/255



trainset = Dataloader(train_matrix,train_label)



## Create Test Set



test_label = test_data['label'].values

test_matrix = test_data.drop('label',axis=1).values/255

testset = Dataloader(test_matrix,test_label)
## Display the images

display_loader = torch.utils.data.DataLoader(trainset,batch_size=50)

batch = next(iter(display_loader))

images,label = batch

import matplotlib.pyplot as plt

grid = torchvision.utils.make_grid(images,nrow=10)

plt.figure(figsize=(15,15))

plt.imshow(np.transpose(grid,(1,2,0)))

plt.show()
## Declare a network object



class Network(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 3,kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels= 3 , out_channels = 6,kernel_size = 5)

        self.fc1 = nn.Linear(in_features=4*4*6,out_features=120)

        self.fc2 = nn.Linear(in_features=120,out_features = 60)

        self.out = nn.Linear(in_features = 60,out_features=10)

        

    def forward(self,t):

        ## Forward Pass

        ## Input Layer

        t = t

        ## First Convolutional Layer

        t = self.conv1(t)

        t = F.relu(t)

        t = F.max_pool2d(t,kernel_size=2,stride=2)

        print('Shape After 1st Convolution and max pooling')

        print(t.shape)

        ## Second Convolutional Layer

        t = self.conv2(t)

        t = F.relu(t)

        t = F.max_pool2d(t,kernel_size=2,stride=2)



        print('Shape After second convolutional network')

        print(t.shape)

        

        ## Flattened hidden layer

        t = t.reshape(-1,4*4*6)

        t = self.fc1(t)

        t = F.relu(t)

        print('Shape After 1st FC Network')

        print(t.shape)

        

        ## Second Hidden layer

        

        t = self.fc2(t)

        t = F.relu(t)

        print('Shape After 2nd FC Network')

        print(t.shape)

        

        ## Output Layer

        

        t = self.out(t)

        print('Shape After Last FC Network')

        print(t.shape)

        

        ## Softmax is not used because F.cross_entropy() function implicitly performs the softmax operation.

        

        

        

        

        return(t)

        

def get_num_correct(pred,labels):

    return(pred.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item())        
## Training and Testing with hyper parameter tuning

## Create a list of Hyper Parameters



from itertools import product



parameters = dict(

    lr = [0.01,0.001],

    batch_size = [100,1000],

    shuffle = [True,False]

    

)

param_values = [v for v in parameters.values()]

param_values



for lr,batch_size,shuffle in product(*param_values):

    print(lr,batch_size,shuffle)
## Unpack the parameters

## Run the 

results = {}

import numpy as np

c = 0



for lr,batch_size,shuffle in product(*param_values):

    network = Network()

    train_loader = torch.utils.data.DataLoader(trainset,batch_size = batch_size,shuffle = shuffle)

    test_loader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle = shuffle)

    optimizer = optim.Adam(network.parameters(),lr=lr)

    images, labels = next(iter(train_loader))

    for epoch in range(5):



        print(epoch)

        total_train_loss = 0

        total_train_correct = 0

        total_test_loss = 0

        total_test_correct = 0

        

        for batch in train_loader:

            images,labels = batch

#             print(images.shape)

#             print(labels.shape)

            pred = network(images.float())

#             print(pred.shape)

            loss = F.cross_entropy(pred,labels.argmax(dim=1))

            optimizer.zero_grad() # flush gradients

            loss.backward() # Calculate Gradients

            optimizer.step() # update weight

            total_train_loss = total_train_loss + (loss.item()*images.shape[0]) ## Total Loss

            total_train_correct = total_train_correct + get_num_correct(pred,labels)

            

        if epoch%1==0:

            with torch.no_grad():

                for batch in test_loader:

                    images,labels = batch

                    pred = network(images.float())

                    loss = F.cross_entropy(pred,labels.argmax(dim=1))

                    total_test_loss = total_test_loss + (loss.item()*images.shape[0])

                    total_test_correct = total_test_correct + get_num_correct(pred,labels)

            result_dict = {

                'epoch' : epoch,

                'total_train_loss' : total_train_loss,

                'total_train_correct': total_train_correct,

                'total_test_loss' : total_test_loss,

                'total_test_correct': total_test_correct,

                'batch_size': batch_size,

                'lr': lr

            }

            results[c] = result_dict

            c = c+1

    

    
correct_train = []

max_index = 0

max_test_accuracy = 0

for i in range(8):

    ## Get the index with the highest test accuracy and 

    if results[i]['total_test_correct'] > max_test_accuracy:

        max_index = i

        max_test_accuracy = results[i]['total_test_correct']

    

    correct_train.append(results[i]['total_test_correct'])



import matplotlib.pyplot as plt

plt.plot(correct_train)

plt.xlabel('')

plt.ylabel('Test Accuracy')

plt.show()





correct_train = []

for i in range(8):

    correct_train.append(results[i]['total_train_correct'])



import matplotlib.pyplot as plt

plt.plot(correct_train)

plt.xlabel('')

plt.ylabel('Train Accuracy')

plt.show()


results[max_index]
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_img = test.iloc[:,1:].astype('float').values/255.0

testset = Dataloader(test_img,None,is_train=False)

test_loader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)

## Re create Model with highest Test score

network = Network()

train_loader = torch.utils.data.DataLoader(trainset,batch_size = results[max_index]['batch_size'],shuffle = shuffle)

optimizer = optim.Adam(network.parameters(),lr=results[max_index]['lr'])

for epoch in range(5):

    for batch in train_loader:

        images,labels = batch

        pred = network(images.float())

        loss = F.cross_entropy(pred,labels.argmax(dim=1))

        optimizer.zero_grad() # flush gradients

        loss.backward() # Calculate Gradients

        optimizer.step() # update weight











predictions = []



for batch in test_loader:

    images = batch

    preds = network(images.float())

    predictions += list(preds.argmax(dim=1).cpu().detach().numpy())

submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv('submission.csv', index=False)

for grp in optimizer.param_groups:

    print(grp['lr'])