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
import pandas as pd
import numpy as np
import random

#split dataset
from sklearn.model_selection import train_test_split

#precision,recall,f1-score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#visualize results
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
#global variables
#seed
np.random.seed(123)
torch.manual_seed(123)
random.seed(123)
data=pd.read_csv("../input/creditcardfraud/creditcard.csv")
data.head(10)
X = data.drop(['Time',"Class"], axis=1)
y = data["Class"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=False)

print(f'Positive Samples in Training Set:{sum(y_train)}')
print(f'Positive Samples in Testing Set:{sum(y_test)}')
#归一化
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler().fit(X_train)

X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#训练集中只包含negative samples
#注意此处归一化与提取negative samples的先后顺序
print(f'Before Extract-X_train Shape:{X_train.shape}')
print(f'Before Extract-y_train Length:{len(y_train)}')

X_train=X_train[y_train==0]
y_train=y_train[y_train==0]

print(f'After Extract-X_train Shape:{X_train.shape}')
print(f'After Extract-y_train Length:{len(y_train)}')
batch_size=128

#transform numpy to pytorch tensor
X_train_tensor=torch.from_numpy(X_train).float()
X_test_tensor=torch.from_numpy(X_test).float()

#fitting by batches (using dataloader)
X_train_dataloader=DataLoader(X_train_tensor,batch_size=batch_size,shuffle=False,drop_last=True)
class generator(nn.Module):
  def __init__(self,input_size,act_fun,deep=False):
    super(generator,self).__init__()

    #deeper neural network or not?
    if not deep:
      hidden_size=input_size//2

      self.encoder_1=nn.Sequential(
        nn.Linear(input_size,hidden_size),
        act_fun
        )

      self.decoder_1=nn.Sequential(
        nn.Linear(hidden_size,input_size)
        )

      self.encoder_2=nn.Sequential(
        nn.Linear(input_size,hidden_size),
        act_fun
        )
      
    elif deep:
      hidden_size_1=input_size//2
      hidden_size_2=hidden_size_1//2

      self.encoder_1=nn.Sequential(
        nn.Linear(input_size,hidden_size_1),
        act_fun,
        nn.Linear(hidden_size_1,hidden_size_2),
        act_fun
        )

      self.decoder_1=nn.Sequential(
        nn.Linear(hidden_size_2,hidden_size_1),
        act_fun,
        nn.Linear(hidden_size_1,input_size)
        )

      self.encoder_2=nn.Sequential(
        nn.Linear(input_size,hidden_size_1),
        act_fun,
        nn.Linear(hidden_size_1,hidden_size_2),
        act_fun
        )

  def forward(self,input):
    z=self.encoder_1(input)
    X_hat=self.decoder_1(z)
    z_hat=self.encoder_2(X_hat)
    return z,X_hat,z_hat
class discriminator(nn.Module):
  def __init__(self,input_size,act_fun,deep=False):
    super(discriminator,self).__init__()
    
    if not deep:
      hidden_size=input_size//2

      self.encoder=nn.Sequential(
        nn.Linear(input_size,hidden_size),
        act_fun
        )

      self.classifier=nn.Sequential(
        nn.Linear(hidden_size,1),
        nn.Sigmoid()
        )
    
    elif deep:
      hidden_size_1=input_size//2
      hidden_size_2=hidden_size_1//2

      self.encoder=nn.Sequential(
        nn.Linear(input_size,hidden_size_1),
        act_fun,
        nn.Linear(hidden_size_1,hidden_size_2),
        act_fun
        )

      self.classifier=nn.Sequential(
        nn.Linear(hidden_size_2,1),
        nn.Sigmoid()
        )
      
  def forward(self,input):
    latent_vector=self.encoder(input)
    output=self.classifier(latent_vector)

    return latent_vector,output
net_generator=generator(input_size=X_train_tensor.size(1),act_fun=nn.Tanh(),deep=True)
net_discriminator=discriminator(input_size=X_train_tensor.size(1),act_fun=nn.Tanh(),deep=False)
print(net_generator)
print(net_discriminator)
L1_criterion=nn.L1Loss()
L2_criterion=nn.MSELoss()

BCE_criterion=nn.BCELoss()
optimizer_G=torch.optim.SGD(net_generator.parameters(),lr=0.01,momentum=0.5)
optimizer_D=torch.optim.SGD(net_discriminator.parameters(),lr=0.01,momentum=0.5)
running_loss_G_vis=np.array([])
running_loss_D_vis=np.array([])

for epoch in range(20):
  running_loss_G=0
  running_loss_D=0

  for X in X_train_dataloader:
    #training the discriminator with real sample
    net_discriminator.zero_grad()

    X=Variable(X)
    y_real=torch.FloatTensor(batch_size).fill_(0)#real label=0,size=batch_size
    y_real=Variable(y_real)

    _,output=net_discriminator(X)

    loss_D_real=BCE_criterion(output,y_real)

    #training the discriminator with fake sample
    _,X_hat,_=net_generator(X)
    y_fake=torch.FloatTensor(batch_size).fill_(1)#fake label=1,size=batch_size

    _,output=net_discriminator(X_hat)

    loss_D_fake=BCE_criterion(output,y_fake)

    #entire loss in discriminator
    loss_D=loss_D_real+loss_D_fake
    running_loss_D+=loss_D
    loss_D.backward()

    optimizer_D.step()

    #training the generator based on the result from the discriminator
    net_generator.zero_grad()

    z,X_hat,z_hat=net_generator(X)

    #loss_1
    latent_vector_real,_=net_discriminator(X)
    latent_vector_fake,_=net_discriminator(X_hat)

    loss_G_1=L2_criterion(latent_vector_fake,latent_vector_real)
    
    #loss_2
    loss_G_2=L1_criterion(X,X_hat)

    #loss_3
    loss_G_3=L1_criterion(z,z_hat)

    #entire loss in generator
    loss_G=loss_G_1+loss_G_2+loss_G_3
    running_loss_G+=loss_G
    loss_G.backward()

    optimizer_G.step()

  running_loss_G=running_loss_G/len(X_train_dataloader)
  running_loss_D=running_loss_D/len(X_train_dataloader)

  running_loss_G_vis=np.append(running_loss_G_vis,running_loss_G.detach().numpy())
  running_loss_D_vis=np.append(running_loss_D_vis,running_loss_D.detach().numpy())

  print(f'Generator Loss in Epoch {epoch}: {running_loss_G:.{4}}')
  print(f'Discriminator Loss in Epoch {epoch}: {running_loss_D:.{4}}\n')
plt.plot(running_loss_G_vis,'b',label='Generator Loss',linewidth=2)
plt.plot(running_loss_D_vis,'r',label='Discriminator Loss',linewidth=2)

plt.title('Generator and Discriminator Loss in Training')
plt.legend()
plt.show()
#testing
#the scores of test samples
score=np.array([])
for i in range(len(X_test_tensor)):
  z,_,z_hat=net_generator(X_test_tensor[i])
  score=np.append(score,L1_criterion(z,z_hat).detach().item())
for ratio in [0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004]:
  y_pred=np.repeat(0,len(y_test))

  y_pred[np.argsort(-score)[:round(ratio*len(y_test))]]=1

  print(f'Precision: {precision_score(y_pred,y_test):.{4}}')
  print(f'Recall: {recall_score(y_pred,y_test):.{4}}')
  print(f'F1-score: {f1_score(y_pred,y_test):.{4}}\n')
threshold_vis=0.001
threshold_score=score[np.argsort(-score)][round(threshold_vis*len(y_test))]

plt.scatter(range(len(score)),score,c='b',label='Normal')
plt.scatter(np.array(range(len(score)))[y_test==1],score[y_test==1],c='r',label='Anomaly')

plt.hlines(threshold_score,xmin=0,xmax=len(score),colors='y',linewidth=4)

plt.title(f'Threshold = {threshold_vis}')
plt.legend()
plt.show()