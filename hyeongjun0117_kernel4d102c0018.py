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

import pandas as pd

import torch.optim as optim

import numpy as np

torch.manual_seed(1)

device = torch.device("cuda")

#데이터

xy_data = pd.read_csv('../input/2020-ai-exam-facepca/2020.AI.facePCA-train.csv')

# xy_data = xy_data.dropna()

x_test = pd.read_csv('../input/2020-ai-exam-facepca/2020.AI.facePCA-test.csv')

submit = pd.read_csv('../input/2020-ai-exam-facepca/2020.AI.face-sample-submission.csv')
xy_data = np.array(xy_data)

x_train = torch.FloatTensor(xy_data[:,1:-1]).to(device)

y_train = torch.LongTensor(xy_data[:,-1]).to(device)



x_test = np.array(x_test)

x_test = torch.FloatTensor(x_test[:,1:]).to(device)
print(x_test.shape)

print(x_train.shape)
W = torch.zeros((150,7)).to(device).detach().requires_grad_(True)

b = torch.zeros(1).to(device).detach().requires_grad_(True)

optimizer= optim.SGD([W,b], lr=1e-1)

nb_epochs = 100
nb_epochs = 10
import torch.nn.functional as F

for epoch in range(nb_epochs + 1):

  hypothesis = F.softmax(x_train.matmul(W)+b,dim=1)

  cost = F.cross_entropy(hypothesis, y_train)

  optimizer.zero_grad()

  cost.backward()

  optimizer.step()

  if epoch%10==0:

    print('cost = {}'.format(cost.item())) #1.25 0.82608 #1.22 0.82919 # 1.3 0.82919
hypothesis = F.softmax(x_test.matmul(W)+b,dim=1)

predict = torch.argmax(hypothesis,dim=1)

for i in range(len(predict)):

  submit['Category'][i]=predict[i]

x_test.shape

submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)