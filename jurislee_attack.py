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



import torch

import torchvision.datasets as data

import torchvision.transforms as transforms

import random



from sklearn import preprocessing
device = 'cuda' if torch.cuda.is_available() else 'cpu'



random.seed(777)

torch.manual_seed(777)

if device == 'cuda':

  torch.cuda.manual_seed_all(777)
# 학습 파라미터 설정

learning_rate = 0.01

training_epochs = 40

batch_size = 100



scaler = preprocessing.MinMaxScaler()



# 데이터의 분포가 다 다르다. 이를 preprocessing.StandardScaler()를 이용해  Scaler를 맞춰주었다. 
# 학습 데이터 로드------------------------------------------

train_data = pd.read_csv('/kaggle/input/2020-abalone-age/2020-abalone-train.csv', header=None, skiprows=1)



x_data = np.array(train_data.loc[:,2:7])

x_data= scaler.fit_transform(x_data)

y_data = np.array(train_data[[8]])



x_data = torch.FloatTensor(x_data)

y_data = torch.FloatTensor(y_data)

# print(x_data)



# 테스트 데이터 로드------------------------------------------



# 데이터 로더--------------------------------------------

train_dataset = torch.utils.data.TensorDataset(x_data, y_data)

data_loader = torch.utils.data.DataLoader(dataset= train_dataset,

                                          batch_size = batch_size,

                                          shuffle = True,

                                          drop_last = True)



print(x_data.shape)
m = 512

m2 = 256

m3 = 128

linear1 =torch.nn.Linear(x_data.shape[1],256, bias = True)

linear2 =torch.nn.Linear(256,256,bias = True)

linear3 =torch.nn.Linear(256,256,bias = True)

linear4 =torch.nn.Linear(256,256,bias = True)

linear5 =torch.nn.Linear(256,1,bias = True)

relu = torch.nn.LeakyReLU()
torch.nn.init.kaiming_uniform_(linear1.weight)

torch.nn.init.kaiming_uniform_(linear2.weight)

torch.nn.init.kaiming_uniform_(linear3.weight)

torch.nn.init.kaiming_uniform_(linear4.weight)

torch.nn.init.kaiming_uniform_(linear5.weight)
model = torch.nn.Sequential(linear1, relu, #dropout

                            linear2, relu, #dropout

                            linear3, relu, #dropout

#                            linear4, relu, #dropout

                            linear5

                            ).to(device)





loss = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
total_batch = len(data_loader)



for epoch in range(training_epochs):

    avg_cost = 0

    for X,Y in data_loader:

        X = X.to(device)

        Y = Y.to(device)

    

        optimizer.zero_grad()

        hypothesis = model(X)

        # cost 계산



        cost = loss(hypothesis, Y)



        # error계산

        cost.backward()

        optimizer.step()



        #평균 에러 계산

        avg_cost +=cost/total_batch



    print('Epoch {:4d}, Cost: {:.6f}'.format(epoch, cost.item()))

print('Learning end')


with torch.no_grad():

    x_test = pd.read_csv('/kaggle/input/2020-abalone-age/2020-abalone-test.csv', header=None)

    x_test = np.array(x_test.loc[:,2:7]) 

    x_test= scaler.transform(x_test)

    x_test = torch.from_numpy(x_test).float().to(device)

    print(x_test)

    prediction = model(x_test)

    

correction_prediction = prediction.cpu().numpy().reshape(-1,1)

submit = pd.read_csv('/kaggle/input/2020-abalone-age/2020-abalone-submit.csv')





# print(submit)



for i in range(len(correction_prediction)):

    submit['Predict'][i] = correction_prediction[i].item()

print(submit)
submit.to_csv('submit.csv')