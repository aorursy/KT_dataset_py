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
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.info()
df.isnull().sum()
df.drop(['id',df.columns[-1]],axis=1,inplace=True)
df.isnull().sum()
df['diagnosis']=df['diagnosis'].astype('category').cat.as_ordered()
df.corrwith(df['diagnosis'].cat.codes)
df.info()
Y=df['diagnosis'].cat.codes
X=df[df.columns[1:-1]]
X=(X-X.mean())/X.std()
X
X.shape
import torch as T
X=T.from_numpy(X.to_numpy())
X=X.float()
Y=T.from_numpy(Y.to_numpy().copy())
import torch.nn as nn

class Network(nn.Module):

    def __init__(self,i_dim,o_dim):

        super(Network,self).__init__()

        self.layer1=nn.Sequential(

                    nn.Linear(i_dim,128),

                    nn.ReLU(),

                    nn.Linear(128,64),

                    nn.Tanh()

        

        

        )

        self.layer2=nn.Sequential(

                    nn.Linear(64,16),

                    nn.ReLU()

                    

        )

        self.layer3=nn.Linear(16,o_dim)

        self.output=nn.LogSoftmax()

        

    def forward(self,X):

        X=self.layer1(X)

        X=self.layer2(X)

        X=self.layer3(X)

        return self.output(X)
net=Network(X.shape[1],2)
print(net)
creterion1=nn.CrossEntropyLoss()

optim=T.optim.Adam(net.parameters())
!pip install torchviz



from torchviz import make_dot
make_dot(net(X))
from sklearn.model_selection import train_test_split as ttt

X_train,X_test,Y_train,Y_test=ttt(X,Y,test_size=0.1)
Y_train
losses=[]

for e in range(1000):

    optim.zero_grad()

    y_hat=net.forward(X_train.float())

    loss=creterion1(y_hat,Y_train.long())

    loss.backward()

    optim.step()

    if e%10==0:

        losses.append(loss.item())

        print('loss at epoch {} is {}'.format(e,loss.item()))
import matplotlib.pyplot as plt

plt.scatter(range(len(losses)),losses)
def predict(model,X):

    y_hat=model.forward(X)

    y_hat=T.exp(y_hat)

    return y_hat.max(axis=1)
y_hat=predict(net,X_test)

y_hat
acc=sum(y_hat[1]==Y_test).float()/len(y_hat[1])
acc
from sklearn.metrics import confusion_matrix as cm
cm(y_hat[1].numpy(),Y_test.numpy())