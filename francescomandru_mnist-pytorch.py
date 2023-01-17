import numpy as np

import matplotlib.pyplot as plt

import scipy.io

import pandas as pd



import torch

import torch.nn as nn

import torch.optim as optim

from torch.autograd import Variable

from torchvision import transforms, utils

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F



from sklearn.model_selection import train_test_split, ParameterGrid



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device used: ", device)
dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train = np.array(dataset.iloc[:,1:]) / 255

label = np.array(dataset.label)

print('Dataset shape: ', dataset.shape)

print('Train shape: ', train.shape)

print('Label shape: ', label.shape)
plt.title('Image 416')

plt.imshow(train[416].reshape(28,28), cmap='gray')

plt.show()
### GENERAL PARAMETER TO CHANGE

testPercentage = 0.20

n_batch = 32



# Split the dataset in train and test

X_train, X_test, Y_train, Y_test = train_test_split(train, label, test_size = testPercentage, random_state = 1204532)

print('Train and Validation shape: ', (X_train.shape, Y_train.shape))

print('Test shape: ', (X_test.shape, Y_test.shape))
# Tensor transformation

torch_X_train = torch.from_numpy(X_train).view(-1,1,28,28).float()

torch_Y_train = torch.from_numpy(Y_train)



torch_X_test = torch.from_numpy(X_test).view(-1,1,28,28).float()

torch_Y_test = torch.from_numpy(Y_test)





train = torch.utils.data.TensorDataset(torch_X_train,torch_Y_train)

test = torch.utils.data.TensorDataset(torch_X_test,torch_Y_test)





train_loader = torch.utils.data.DataLoader(train, batch_size = n_batch)

test_loader = torch.utils.data.DataLoader(test, batch_size = len(test))
def denseNeuronsNumber(inputDim, k, p, s, kp, conv, pool):

  

  output = 0

  

  for i in range(conv):



    if i==0:

      output = int( (inputDim-k+2*p)/s ) + 1 

    else:

      output = int( (output-k+2*p)/s ) + 1

    if pool==True:

      output = int( (output-kp)/kp ) + 1



    #print('Image size layer',i,': ',output)



  return output
class Net(nn.Module):

    

    def __init__(self, Fm1, Fm2, Ks, pad, stride, pool, Ni, Nh1, No, act, drop_rate):

        # super() lets you avoid referring to the base class explicitly

        super().__init__()



        '''

          in_channels = 1 since image is grayscale 

          out_channels = Number of feature maps you want in output

          act = Activation function chosen

          Fm1 = Feature maps layer conv 1

          Fm2 = Feature maps layer conv 2

          Fm3 = Feature maps layer conv 3

          Ks = Kernel size  

          Ni = Input of Linear Layer 

          Nh1 = Number of neurons for the first hidden dense layer

          No = Output = 10

          act = Activation function

          Droprate = % of dropout 

        '''

        

        self.conv1 = nn.Sequential(

            

            nn.Conv2d(in_channels=1, out_channels=Fm1, kernel_size=Ks, 

                      stride=stride, padding=pad),

            act_dict[act],

            nn.BatchNorm2d(Fm1),

            nn.MaxPool2d(kernel_size=pool),

            nn.Dropout(drop_rate)

        )



        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=Fm1, out_channels=Fm2, kernel_size=Ks, 

                      stride=stride, padding=pad),

            act_dict[act],

            nn.BatchNorm2d(Fm2),

            nn.MaxPool2d(kernel_size=pool),

            nn.Dropout(drop_rate)

        )

        

        self.fc1 = nn.Sequential(

            nn.Linear(Ni, Nh1),

            act_dict[act],

            nn.BatchNorm1d(Nh1)

        )

        

        self.fc2 = nn.Sequential(

           nn.Linear(Nh1,No) 

        )

              

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        

        #print("X prev: ", x.shape)

        x = x.view(x.size(0),-1)

        #print("X after: ", x.shape)

        x = self.fc1(x)

        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)

        

        return x
act_dict = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'elu': nn.ELU()}



Fm1 = 16

Fm2 = 32

Ks = 3 

pad = 0

pool = 2

stride = 1 

Ni = denseNeuronsNumber(28, Ks, pad, stride, pool, 2, True)

Ni = Ni * Ni * Fm2

Nh1 = int(Ni / 2)

No = 10

act = 'relu'

drop_rate = 0.20



model = Net(Fm1, Fm2, Ks, pad, stride, pool, Ni, Nh1, No, act, drop_rate)

criterion = nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)





############### FINAL TRAINING TEST ################



epochs = 100

loss_train_list = []

loss_test_list = []



acc_train_list = []

acc_test_list = []



model.to(device)



for epoch in range(epochs):



  loss_train = 0

  loss_test = 0



  total_train = total_test = correct_train = correct_test = 0



  model = model.train()



  for n_batch, (images,labels) in enumerate(train_loader):



    optimizer.zero_grad()

    images = images.to(device)

    labels = labels.to(device)

    outputs = model(images)

    loss = criterion(outputs,labels)

    loss.backward()

    optimizer.step()





  model = model.eval()



  with torch.no_grad():



    for n_batch, (images_train,labels_train) in enumerate(train_loader):



      images_train = images_train.to(device)

      labels_train = labels_train.to(device)

      outputs_train = model(images_train)

      lossT = criterion(outputs_train,labels_train)

      loss_train += lossT.item()

      # Accuracy Prediction

      _, predicted = torch.max(outputs_train, 1)

      total_train += labels_train.size(0)

      correct_train += (predicted == labels_train).sum().item()



    for _, (images_test, labels_test) in enumerate(test_loader):

        

      images_test = images_test.to(device)

      labels_test = labels_test.to(device)

      outputs_test = model(images_test)

      lossS = criterion(outputs_test,labels_test)

      loss_test += lossS.item()

      # Accuracy Prediction

      _, predicted = torch.max(outputs_test, 1)

      total_test += labels_test.size(0)

      correct_test += (predicted == labels_test).sum().item()



  acc_train_list.append( correct_train / total_train )

  acc_test_list.append( correct_test / total_test )  

  

  loss_train_list.append(loss_train / len(train_loader))

  loss_test_list.append(loss_test / len(test_loader)) 





    

  print(' Epoch [{}/{}], Training Loss: {:.4f}, Test Loss: {:.4f}'

    .format(epoch+1, epochs, loss_train_list[epoch],

      loss_test_list[epoch]))

  

  print(' Epoch [{}/{}], Training Accuracy: {:.4f}, Test Accuracy: {:.4f}'

    .format(epoch+1, epochs, acc_train_list[epoch],

      acc_test_list[epoch]))

  

  print('\n---')
testset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

torch_test = torch.from_numpy(np.array(testset)/255).view(-1,1,28,28).float()
predictions = []

model.to('cuda')

model.eval()

with torch.no_grad():

    for test in torch_test:

        t = test.to('cuda')

        #print('t:', t)

        prediction = model(t.reshape(1,1,28,28))

        #print('prediction before: ', prediction)

        _, prediction = torch.max(prediction,1)

        #print('prediction after: ', prediction)

        predictions.append(prediction)
predictions = [p.item() for p in predictions]
lista = np.arange(1,28001,1)



my_submission = pd.DataFrame({'ImageId': lista, 'Label': predictions})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)