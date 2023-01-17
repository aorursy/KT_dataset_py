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

import torch.nn.functional as F

import torch.optim as optim

import numpy as np

import pandas as pd

import random

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler
device = 'cuda' if torch.cuda.is_available() else 'cpu'



random.seed(77)

torch.manual_seed(77)

if device == 'cuda':

  torch.cuda.manual_seed_all(77)
train = pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-2/mnist_train_label.csv',header=None)

test = pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-2/mnist_test.csv',header=None)
# 학습 파라미터 설정

learning_rate = 0.001

training_epochs = 15

batch_size = 100
Scaler = preprocessing.Normalizer()
x_train_data=train.loc[:,1:784]

y_train_data=train.loc[:,0]



x_train_data=np.array(x_train_data)

y_train_data=np.array(y_train_data)



x_train_data=Scaler.fit_transform(x_train_data)

x_train_data=torch.FloatTensor(x_train_data)

y_train_data=torch.LongTensor(y_train_data)
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)



data_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                          batch_size=batch_size,

                                          shuffle=True,

                                          drop_last=True)

linear1 = torch.nn.Linear(784,10,bias=True)
torch.nn.init.xavier_uniform_(linear1.weight)

model = torch.nn.Sequential(linear1).to(device)

loss = torch.nn.CrossEntropyLoss().to(device) 

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate,momentum=0.9)
total_batch = len(data_loader)

for epoch in range(training_epochs):

    avg_cost = 0



    for X, Y in data_loader:



        # (1000, 1, 28, 28) 크기의 텐서를 (1000, 784) 크기의 텐서로 변형

        X = X.view(-1, 28 * 28).to(device)

        # one-hot encoding되어 있지 않음

        Y = Y.to(device)

        #%debug



        # 그래디언트 초기화

        optimizer.zero_grad()

        # Forward 계산

        hypothesis = model(X)

        # Error 계산

        cost = loss(hypothesis, Y)

        # Backparopagation

        cost.backward()

        # 가중치 갱신

        optimizer.step()



        # 평균 Error 계산

        avg_cost += cost / total_batch



    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))



print('Learning finished')
with torch.no_grad():



    x_test_data=test.loc[:,:]

    x_test_data=np.array(x_test_data)

    x_test_data=torch.from_numpy(x_test_data).float().to(device)



    prediction = model(x_test_data)

    correct_prediction = torch.argmax(prediction, 1)
correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)

submit=pd.read_csv('/kaggle/input/2020-ai-exam-fashionmnist-2/submission.csv')
for i in range(len(correct_prediction)):

    submit['Category'][i]=correct_prediction[i].item()
submit.to_csv('baseline.csv',index=False,header=True)