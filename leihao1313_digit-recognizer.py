# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pickle



import sklearn

import sklearn.preprocessing



import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

from torchvision.transforms import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



X_train = df.iloc[:,1:].to_numpy()/255.0

# X_train = df.iloc[:,1:].to_numpy()

y_train = df.iloc[:,0].to_numpy()

X_test = df_test.to_numpy()/255.0



X_train = X_train.reshape(-1,1,28,28)

X_test = X_test.reshape(-1,1,28,28)

X_train = torch.tensor(X_train.astype(np.float32))

X_test = torch.tensor(X_test.astype(np.float32))

y_train = torch.tensor(y_train.astype(np.int64))
# tranform functions

T = Compose([

    ToPILImage(),

    RandomCrop(24, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),

    RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3),

    Pad(2, fill=0, padding_mode='constant'),

    RandomRotation(30, resample=False, expand=False, center=None),

#     CenterCrop(24),

#     Grayscale(1),

#     RandomResizedCrop(28, scale=(0.5, 1), ratio=(0.75, 1), interpolation=2),

#     ColorJitter(brightness=1, contrast=0, saturation=0, hue=0),

    ToTensor(),

#     Normalize((0.5,), (0.5,)),

#     ToPILImage(),

    ])
X_aug = torch.stack([T(x) for x in X_train])
plt.figure(figsize=(20,100))

for i,x in enumerate(X_aug[100:110]):

    plt.subplot(1,10,i+1)

    plt.imshow(x.reshape(28,28),cmap='binary')

      
plt.figure(figsize=(20,100))

for i,x in enumerate(X_train[100:110]):

    plt.subplot(1,10,i+1)

    plt.imshow(x.reshape(28,28),cmap='binary')  
X_train = torch.stack([X_aug,X_train]).reshape(-1,1,28,28)

y_train = torch.stack([y_train,y_train]).reshape(-1,)
image_channels = X_train.shape[1]

image_width = X_train.shape[2]

num_filters = 32

num_filters2 = 64

num_filters3 = 128

filter_size = 5

pool_size = 2

# final_input = (((image_width+1-filter_size)//pool_size+1-filter_size)//pool_size)**2*num_filters2#without padding

final_input = (image_width//pool_size//pool_size//pool_size)**2*num_filters3#with padding



model = torch.nn.Sequential(

    torch.nn.Conv2d(image_channels, num_filters, filter_size, padding=filter_size//2),

    torch.nn.ReLU(),

    torch.nn.MaxPool2d(pool_size, pool_size),

    

    torch.nn.Conv2d(num_filters, num_filters2, filter_size, padding=filter_size//2),

    torch.nn.ReLU(),

    torch.nn.MaxPool2d(pool_size, pool_size),

    

    torch.nn.Conv2d(num_filters2, num_filters3, filter_size, padding=filter_size//2),

    torch.nn.ReLU(),

    torch.nn.MaxPool2d(pool_size, pool_size),

    

    torch.nn.Flatten(),

    torch.nn.Linear(final_input, final_input//2),

    torch.nn.ReLU(),

    nn.Dropout(p=0.2),

    

    torch.nn.Linear(final_input//2, final_input//4),

    torch.nn.ReLU(),

    nn.Dropout(p=0.3),

    

    torch.nn.Linear(final_input//4, final_input//16),

    torch.nn.ReLU(),

    nn.Dropout(p=0.5),

    

    torch.nn.Linear(final_input//16,10),

)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Assuming that we are on a CUDA machine, this should print a CUDA device:



print(device)

model.to(device)



def train_models(batch_size,num_epoch,train_loader):



    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)



    #mini-batch training loop

    for epoch in range(num_epoch):



        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            X, y = data

            X, y = X.to(device), y.to(device)



            optimizer.zero_grad()

#             optimizer2.zero_grad()



            # forward + backward + optimize



            y_pred = model(X)



            loss = criterion(y_pred, y)

            loss.backward()

            optimizer.step()



            #avg batch loss

            running_loss += loss.item()

            batch = X_train.shape[0]//batch_size//3

            if (i+1) % batch == 0:

                print('Epoch: %d, Batch: %5d had avg loss: %.4f'%(epoch+1,i+1,running_loss/batch))

                running_loss = 0.0

        if (epoch+1) % 10 ==0:

            torch.save(model.state_dict(), 'model_%s.pt'%(epoch+1))

    torch.save(model.state_dict(), 'model.pt')

    print('Training Finished')

    return model
batch_size=128

num_epoch=1

# create torch Dataset class from tensors

train_set = torch.utils.data.TensorDataset(X_train, y_train)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)



try:#try to load trained model 

    model.load_state_dict(torch.load('model.pt'))

    print("Trained model loaded")

except:

    print('Model not found, start training...')

    model = train_models(batch_size,num_epoch,train_loader)
# num_epoch=100

# model = train_models(batch_size,num_epoch,train_loader)
test_set = torch.utils.data.TensorDataset(X_test)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
# with open('results.csv','bw') as file:

#     file.write("ImageId,Label\n".encode())

#     with torch.no_grad():

#         for i, data in enumerate(test_loader, 1):

#             X = data[0]

#             X = X.to(device)

#             y_proba = model(X)

#             _, y_pred = torch.max(y_proba.data, 1)

# #             print(y_pred.item())

#             pair = '%s,%s\n'%(i,y_pred.item())

# #             print(pair)

#             file.write(pair.encode())
print('finish')