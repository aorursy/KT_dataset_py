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
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing
train=pd.read_csv('/kaggle/input/carclassification/car5_train.csv')
x_train=train.loc[:,[i for i in train.keys()[1:-1]]]
y_train=train[train.keys()[-1]]
x_train
y_train
Scaler=preprocessing.StandardScaler()
x_train=Scaler.fit_transform(x_train)
x_train=np.array(x_train)
y_train=np.array(y_train)
x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)
x_train
import torch.nn.functional as F
import torch.optim as optim

nb_class=8
nb_data=len(y_train)
W=torch.zeros((8,nb_class),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

optimizer=torch.optim.SGD([W,b],lr=0.5)

nb_epochs=1000


for epoch in range(nb_epochs+1):
  hypothesis=F.softmax(x_train.matmul(W)+b,dim=1)
  cost = F.cross_entropy((x_train.matmul(W) + b), y_train)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch%100==0:
    print('Epoch: {:4d}/{}  Cost: {:.6f}'.format(epoch,nb_epochs,cost.item()))
hypothesis=F.softmax(x_train.matmul(W)+b,dim=1)
predict=torch.argmax(hypothesis,dim=1)

print(predict)
print(y_train)

correct_prediction=predict.float()==y_train
print(correct_prediction)

accuracy=correct_prediction.sum().item()/len(correct_prediction)
print('Accuracy: {:2.2f}'.format(accuracy*100))
test=pd.read_csv('/kaggle/input/carclassification/car5_test.csv')
x_test=test.loc[:,[i for i in test.keys()[1:]]]
x_test=Scaler.fit_transform(x_test)
x_test=np.array(x_test)
x_test=torch.FloatTensor(x_test)
hypothesis=F.softmax(x_test.matmul(W)+b,dim=1)
predict=torch.argmax(hypothesis,dim=1)

print(predict.shape)
predict

submit=pd.read_csv('/kaggle/input/carclassification/car5_submit.csv')

submit
predict=predict.detach().reshape(-1,1)

id=np.array([i for i in range(len(predict))]).reshape(-1,1)
result=np.hstack([id,predict])

submit=pd.DataFrame(result,columns=["Id","Category"])
submit.to_csv("baseline.csv",index=False,header=True)
submit