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
#import necessary libraries

import numpy as np

import torch

import pandas as pd

import matplotlib.pyplot as plt

from torchvision import datasets,transforms,models

import torch.nn.functional as F

from torch import nn,optim



from torch.utils.data import Dataset, TensorDataset

from torch.optim import lr_scheduler



import torchvision

import seaborn as sns
train_dir='/kaggle/input/sign-language-mnist/sign_mnist_train.csv'

test_dir='/kaggle/input/sign-language-mnist/sign_mnist_test.csv'



train=pd.read_csv(train_dir)

test=pd.read_csv(test_dir)
#check train data

train.head(10) 
#check test data

test.head(10)
#Number of classes we have 



print(train['label'].unique())



print("Number of classes : ",len(train['label'].unique()))
#obtain all rows and all columns except the 0 index column



train_data = train.iloc[:, 1:].values

print("Number of train images:", train_data.shape[0])

train_labels=train.loc[:, 'label']

print("Number of pixels in each image:", train_data.shape[1])



test_data = test.iloc[:, 1:].values

print("Number of test images:", test_data.shape[0])

test_labels=test.loc[:, 'label']

print("Number of pixels in each image:", test_data.shape[1])







new_train_labels=np.where(train_labels>8, train_labels-1, train_labels)

new_test_labels=np.where(test_labels>8, test_labels-1, test_labels)







unique_val = np.array(new_test_labels)

#np.append (unique_val, 9)

print(np.unique(unique_val))

train_data.shape , new_train_labels.shape
from PIL import Image 



Image.open("/kaggle/input/sign-language-mnist/amer_sign2.png")
letters={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'k',10:'L',11:'M',12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',19:'U',20:'V',21:'W',22:'X',23:'Y'}


plt.figure(figsize=(24,8))



for i in range(27):

  

  plt.subplot(3,9,i+1)

  plt.imshow(train_data[i].reshape(28,28))

  plt.axis('off')

  plt.title(letters[int((new_train_labels[i]))])


plt.figure(figsize = (30,10))

sns.countplot(new_train_labels)


#Data Augmentation



data_transforms = {

    'train': transforms.Compose([

        transforms.ToPILImage(),

        transforms.RandomRotation(25),

        

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.5], 

                            [0.5])

]),

    

    'valid': transforms.Compose([

        transforms.ToPILImage(),

        transforms.ToTensor(),

        transforms.Normalize([0.5], 

                             [0.5])

    ]),

}
class DtProcessing(Dataset):

    

    #initialise the class variables - transform, data, target

    def __init__(self, data, target, transform=None): 

        self.transform = transform

        self.data = data.reshape((-1,28,28)).astype(np.float32)[:,:,:,None]

        # converting target to torch.LongTensor dtype

        self.target = torch.from_numpy(target).long() 

    

    #retrieve the X and y index value and return it

    def __getitem__(self, index): 

        return self.transform(self.data[index]), self.target[index]

    

    #returns the length of the data

    def __len__(self): 

        return len(list(self.data))

      

#divide train set into train and validation set



from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(train_data, new_train_labels, test_size = .2, random_state = 42)
      

dset_train = DtProcessing(X_train, y_train, transform=data_transforms['train'])



train_loader = torch.utils.data.DataLoader(dset_train, batch_size=20,

                                          shuffle=True, num_workers=0)



dset_valid = DtProcessing(X_valid, y_valid, transform=data_transforms['valid'])



valid_loader = torch.utils.data.DataLoader(dset_valid, batch_size=20,

                                          shuffle=True, num_workers=0)





dset_test = DtProcessing(test_data, new_test_labels, transform=data_transforms['valid'])



test_loader =torch.utils.data.DataLoader(dset_test, batch_size=32, shuffle=True)


import torch.nn.functional as F





class Net(nn.Module):

  

  def __init__(self):

    

    super(Net,self).__init__()

    

    #input depth , output depth , kernel size(filter)x

    

    self.conv1=nn.Conv2d(1,32,kernel_size=(3, 3),padding=(1, 1),stride=(1, 1))

    

    self.conv2=nn.Conv2d(32,32,kernel_size=(3, 3),padding=(1, 1),stride=(1, 1))

    

    self.conv3=nn.Conv2d(32,64,kernel_size=(3, 3),padding=(1, 1),stride=(1, 1))

    

    #padding for last conv layer 

    self.adapt = nn.AdaptiveMaxPool2d((3,3))  

    

    #padding layer

    self.pool=nn.MaxPool2d(2,2)

    

    #dropout layer

    self.drop=nn.Dropout(p=0.2)

    

    #fc layers 

    self.fc1=nn.Linear(64*3*3,240)

    

   

    self.fc2=nn.Linear(240,24)

    

    self.softmax = nn.LogSoftmax(dim=1)

    

  def forward(self,x):

    

    x=self.pool(F.leaky_relu(self.conv1(x)))

    

    x=self.pool(F.leaky_relu(self.conv2(x)))

    

    x=self.adapt(F.leaky_relu(self.conv3(x)))

    

    

    #flatten Images

    x = x.view(x.size(0), -1)

    

    x=self.drop(x)

    

    x=F.leaky_relu(self.fc1(x))

    

    

    x=self.drop(x)

    

    x=self.fc2(x)

    

    return self.softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

if torch.cuda.is_available():

  

    model = model.cuda()

    criterion = criterion.cuda()
from torch.autograd import Variable







def train(n_epochs=100):

  

  Training_loss=[]

  Validation_loss=[]

  

  valid_loss_min = np.Inf # track change in validation loss



  for epoch in range(1, n_epochs+1):

    

    



        # keep track of training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

        #train

        model.train()

    

    

        for data, target in train_loader:

                 

         

        

            if torch.cuda.is_available():

                   data, target = data.cuda(), target.cuda()

            

        # clear the gradients of all optimized variables

            optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

            output = model(Variable(data))

        

        #print(target.shape)

        # calculate the batch loss

            loss = criterion(output, Variable(target))

        # backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

        # perform a single optimization step (parameter update)

            optimizer.step()

        # update training loss

            train_loss += loss.item()*data.size(0)

    #validate

        model.eval()

        accuracy=0.0

        with torch.no_grad():

          for data, target in valid_loader:

      

                 data, target = Variable(data), Variable(target)

        

        #data, target = Variable(data), Variable(target)

                 if torch.cuda.is_available():

                         data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

                 output = model(data)

        

        # calculate the batch loss

                 loss = criterion(output, target)

        # update average validation loss 

                 valid_loss += loss.item()*data.size(0)

        

        

        

    

    # calculate average losses

        train_loss = train_loss/len(train_loader.dataset)

        valid_loss = valid_loss/len(valid_loader.dataset)

        

        Training_loss.append(train_loss/len(train_loader))

        Validation_loss.append(valid_loss/len(valid_loader))

        

        # print training/validation statistics 

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t '.format(

        epoch, train_loss, valid_loss))

    

    

    

    

    

    # save model if validation loss has decreased

        if valid_loss <= valid_loss_min:

                  print("==============================================================================================")

                  print('Validation loss decreased ({:.6f} --> {:.6f}).  >>>>>>>  Saving model ...'.format(

                   valid_loss_min,

                     valid_loss)) 

                  print("==============================================================================================")

                  torch.save(model.state_dict(), 'SignModel1.pt')

                  valid_loss_min = valid_loss

  plt.figure(figsize = (25,10))                

  plt.plot(Training_loss, label='Training loss')

  plt.plot(Validation_loss, label='Validation loss')

  plt.legend(frameon=False)      

        
train(80)
model.load_state_dict(torch.load('SignModel1.pt'))



classes=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
def Cal_accurac():

  test_loss = 0.0

  class_correct = list(0. for i in range(24))

  class_total = list(0. for i in range(24))



  model.eval()

# iterate over test data

  for data, target in test_loader:

    

    batch_size = data.size(0)

    #print(batch_size)

    # move tensors to GPU if CUDA is available

    if torch.cuda.is_available():

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

    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class

    for i in range(batch_size):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# average test loss

  test_loss = test_loss/len(test_loader)

  print('Test Loss: {:.6f}\n'.format(test_loss))



  for i in range(24):

    if class_total[i] > 0 :

        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (

            classes[i], 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))



  print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
Cal_accurac()