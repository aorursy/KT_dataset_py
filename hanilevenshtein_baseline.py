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

from sklearn.preprocessing import MinMaxScaler

import torch

import torch.optim as optim

import torch.nn.functional as F

train=pd.read_csv("/kaggle/input/aqiprediction/train.csv")



print(train["PM10"].value_counts(normalize=True))



train['시간'] = train['시간'].astype(str)

train['시간'] = pd.to_datetime(train['시간'],format="%Y-%m-%d:%H", errors='ignore')

train=train.set_index('시간')



wind=pd.DataFrame({'풍향': np.cos(np.pi*train["풍향"]/360)})

train=train.drop("풍향",axis=1)

train=pd.concat((train,wind),axis=1)



train_x=train[['습도','강수','기온','풍속','풍향']]

train_y=train["PM10"].replace(["좋음","보통","나쁨","매우나쁨"], [0,1,2,3])

scaler=MinMaxScaler()

train_x=scaler.fit_transform(train_x)

train_x=torch.FloatTensor(train_x)

train_y=torch.LongTensor(np.array(train_y))



W1=torch.randn((5,4),requires_grad=True)

b1=torch.randn(4,requires_grad=True)

optimizer1 = optim.SGD([W1,b1],lr=1e-4,momentum=0.9)



nb_epochs=7000



for epoch in range(nb_epochs+1):

  hypothesis1=F.softmax(train_x.matmul(W1)+b1,dim=1)

  cost1=F.cross_entropy((train_x.matmul(W1)+b1),train_y)

  optimizer1.zero_grad()

  cost1.backward()

  optimizer1.step()



  if epoch%1000==0:

    print('Epoch {}/{} Cost1: {:.5f} '.format(

            epoch, nb_epochs, cost1.item()

      ))

hypothesis1=F.softmax(train_x.matmul(W1)+b1,dim=1)

predict1=torch.argmax(hypothesis1,dim=1)

cor1=predict1.float()==train_y

acc1=cor1.sum().item()/len(cor1)

print('{:2.2f}%'.format(acc1*100))

test=pd.read_csv("/kaggle/input/aqiprediction/test.csv",index_col = False).drop("시간", axis = 1) 

wind=pd.DataFrame({'풍향': np.cos(np.pi*test["풍향"]/360)})

test=test.drop("풍향",axis=1)

test=pd.concat((test,wind),axis=1)

test=np.array(test)

test=torch.FloatTensor(test)



hypothesis1 = F.softmax(test.matmul(W1) + b1, dim=1)

predict1=torch.argmax(hypothesis1,dim=1)



submission=pd.read_csv('/kaggle/input/aqiprediction/submission.csv', index_col=False,encoding='utf-8-sig')

for i in range(len(predict1)):

  submission['PM10'][i]=int(predict1[i])



submission['PM10']=submission['PM10'].astype(int)

submission["PM10"]=submission["PM10"].replace([0,1,2,3],["좋음","보통","나쁨","매우나쁨"])
