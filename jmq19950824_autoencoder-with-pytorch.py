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



import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split 

from sklearn.manifold import TSNE

from sklearn import preprocessing 



#precision,recall,f1-score metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
import torch

import torchvision

from torch import nn

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms
#seed

random.seed(123)

np.random.seed(123)

torch.manual_seed(123)
data=pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.head(10)
X = data.drop(['Time','Class'], axis=1)

y = data["Class"].values



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=False)



print(f'Positive Samples in Training Set:{sum(y_train)}')

print(f'Positive Samples in Testing Set:{sum(y_test)}')
#MinMaxScaler

from sklearn.preprocessing import MinMaxScaler



scaler=MinMaxScaler().fit(X_train)



X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)
print(f'Before Extract-X_train Shape:{X_train.shape}')

print(f'Before Extract-y_train Length:{len(y_train)}')



X_train_norm=X_train[y_train==0]

y_train_norm=y_train[y_train==0]



print(f'After Extract-X_train Shape:{X_train_norm.shape}')

print(f'After Extract-y_train Length:{len(y_train_norm)}')
batch_size=128



#transform numpy to pytorch tensor

X_train_tensor=torch.from_numpy(X_train).float()

X_test_tensor=torch.from_numpy(X_test).float()



#fitting by batches (using dataloader)

X_train_dataloader=DataLoader(X_train_tensor,batch_size=batch_size,shuffle=False,drop_last=True)
class autoencoder(nn.Module):

  def __init__(self,input_size,act_fun,deep=False):

    super(autoencoder,self).__init__()



    if not deep:

      hidden_size=input_size//2



      self.encoder=nn.Sequential(

        nn.Linear(input_size,hidden_size),

        act_fun

        )

      

      self.decoder=nn.Sequential(

        nn.Linear(hidden_size,input_size)

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

      

      self.decoder=nn.Sequential(

        nn.Linear(hidden_size_2,hidden_size_1),

        act_fun,

        nn.Linear(hidden_size_1,input_size)

        )

      

  def forward(self,input):

    z=self.encoder(input)

    X_hat=self.decoder(z)



    return z,X_hat
#autoencoder model

model_ae=autoencoder(input_size=X_train_tensor.size(1),act_fun=nn.Tanh(),deep=True)

#loss function

L1_criterion=nn.L1Loss()

#optimizer

optimizer=torch.optim.SGD(model_ae.parameters(),lr=0.01,momentum=0.5)
for epoch in range(20):

  running_loss=0



  for X in X_train_dataloader:

    

    optimizer.zero_grad()



    input=Variable(X)

    z,X_hat=model_ae(input)



    loss=L1_criterion(X,X_hat)

    running_loss+=loss



    loss.backward()

    optimizer.step()    



  running_loss=running_loss/len(X_train_dataloader)



  print(f'AutoEncoder Loss in Epoch {epoch}: {running_loss:.{4}}')
#testing

#the scores of test samples

score=np.array([])



for i in range(len(X_test_tensor)):

  _,X_hat=model_ae(X_test_tensor[i])

  score=np.append(score,L1_criterion(X_test_tensor[i],X_hat).detach().item())
for ratio in [0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004]:



  y_pred=np.repeat(0,len(y_test))



  y_pred[np.argsort(-score)[:round(ratio*len(y_test))]]=1



  print(f'Precision: {precision_score(y_pred,y_test):.{4}}')

  print(f'Recall: {recall_score(y_pred,y_test):.{4}}')

  print(f'F1-score: {f1_score(y_pred,y_test):.{4}}\n')