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
#### train csv 읽어오기

train=pd.read_csv("/kaggle/input/2020-ai-exam-diabetes/2020.AI.diabetes-train.csv")

test=pd.read_csv("/kaggle/input/2020-ai-exam-diabetes/2020.AI.diabetes-test.csv")
#### library import

import os

import pandas as pd

import numpy as np

import torch

from sklearn.preprocessing import normalize
#### train data 와 Label 분리

x_data=train.loc[:,[i for i in train.keys()[:-1]]]

y_data=train[train.keys()[-1]]
#### DataFrame -> Numpy

x_data=np.array(x_data)

y_data=np.array(y_data)

mean=x_data[np.isnan(x_data)==False].mean()

#### x_data 에 NaN 값이 있어 이것들을 0으로 임의로 초기화

#import pdb;pdb.set_trace()

x_data[np.isnan(x_data)]=mean

x_data[x_data==0]=mean

#### Numpy -> Torch Tensor

x_data=normalize(x_data,norm="l1")

x_data=torch.FloatTensor(x_data)

y_data=torch.LongTensor(y_data)
import torch.nn.functional as F 

#### Initialize 

W = torch.zeros((8,2), requires_grad=True, device="cuda")

b = torch.zeros(2, requires_grad=True, device="cuda")

x_data=x_data.cuda()

y_data=y_data.cuda()



optimizer = torch.optim.SGD([W, b], lr=1e-2)

nb_epochs = 10000



for epoch in range(nb_epochs +1):

  

  hy=(x_data.matmul(W) + b)

  

  cost = F.cross_entropy((x_data.matmul(W) + b), y_data) 

  

  optimizer.zero_grad()

  cost.backward()

  optimizer.step()



  if epoch % 5000 == 0:

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
#### Test data Processing

x_test=test.loc[:,[i for i in test.keys()[:]]]

x_test=np.array(x_test)

x_test=normalize(x_test,norm="l1")

x_test=torch.FloatTensor(x_test).cuda()
#### Make Predict

predict=F.softmax((x_test.matmul(W) + b))

predict=torch.argmax(predict,dim=1)
##### Make CSV

predict=predict.detach().cpu().numpy().reshape(-1,1).astype(np.uint32)

id=np.array([i for i in range(predict.shape[0])]).reshape(-1,1).astype(np.uint32)

result=np.hstack([id,predict])

df=pd.DataFrame(result,columns=["id","Label"])

df.to_csv("submission.csv",index=False)