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

import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

import random



if torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')



random.seed(1)

torch.manual_seed(1)

torch.cuda.manual_seed_all(1)
data = pd.read_csv('/kaggle/input/metro/train.csv', encoding='utf-8')

data = data.drop(['역번호', '역이름', '호선', '계', '일평균'], axis = 1)



# 데이터 standardization

mean1 = data[[data.keys()[0]]].mean(axis = 0)

data[[data.keys()[0]]] -= mean1

std1 = data[[data.keys()[0]]].std(axis = 0)

data[[data.keys()[0]]] /= std1



mean2 = data[[data.keys()[1]]].mean(axis = 0)

data[[data.keys()[1]]] -= mean2

std2 = data[[data.keys()[1]]].std(axis = 0)

data[[data.keys()[1]]] /= std2



mean3 = data[[data.keys()[2]]].mean(axis = 0)

data[[data.keys()[2]]] -= mean3

std3 = data[[data.keys()[2]]].std(axis = 0)

data[[data.keys()[2]]] /= std3



mean4 = data[[data.keys()[3]]].mean(axis = 0)

data[[data.keys()[3]]] -= mean4

std4 = data[[data.keys()[3]]].std(axis = 0)

data[[data.keys()[3]]] /= std4



mean5 = data[[data.keys()[4]]].mean(axis = 0)

data[[data.keys()[4]]] -= mean5

std5 = data[[data.keys()[4]]].std(axis = 0)

data[[data.keys()[4]]] /= std5



mean6 = data[[data.keys()[5]]].mean(axis = 0)

data[[data.keys()[5]]] -= mean6

std6 = data[[data.keys()[5]]].std(axis = 0)

data[[data.keys()[5]]] /= std6



mean7 = data[[data.keys()[6]]].mean(axis = 0)

data[[data.keys()[6]]] -= mean7

std7 = data[[data.keys()[6]]].std(axis = 0)

data[[data.keys()[6]]] /= std7



meanY = data[[data.keys()[7]]].mean(axis = 0)

data[[data.keys()[7]]] -= meanY

stdY = data[[data.keys()[7]]].std(axis = 0)

data[[data.keys()[7]]] /= stdY



num_data = data.to_numpy()

x_data = num_data[:,:-1];

y_data = num_data[:,[-1]]



x_data = torch.FloatTensor(x_data).to(device)

y_data = torch.FloatTensor(y_data).to(device)
linear1 = torch.nn.Linear(7, 1, bias=True)

torch.nn.init.xavier_uniform_(linear1.weight)



model = torch.nn.Sequential(linear1).to(device)



loss = torch.nn.MSELoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 18001

for epoch in range(epochs):

  optimizer.zero_grad();

  hypothesis = model(x_data);

  cost = loss(hypothesis, y_data)

  cost.backward();

  optimizer.step()



  if(epoch%1000 == 0):

    print('Epoch: {}, Cost: {}'.format(epoch, cost.item()))
test = pd.read_csv('/kaggle/input/metro/test.csv', encoding='utf-8')

test = test.drop(['역이름', '역번호', '계', '일평균', '호선'], axis=1)



# 데이터 standardization\

test[[test.keys()[0]]] -= mean1

test[[test.keys()[0]]] /= std1



test[[test.keys()[1]]] -= mean2

test[[test.keys()[1]]] /= std2



test[[test.keys()[2]]] -= mean3

test[[test.keys()[2]]] /= std3



test[[test.keys()[3]]] -= mean4

test[[test.keys()[3]]] /= std4



test[[test.keys()[4]]] -= mean5

test[[test.keys()[4]]] /= std5



test[[test.keys()[5]]] -= mean6

test[[test.keys()[5]]] /= std6



test[[test.keys()[6]]] -= mean7

test[[test.keys()[6]]] /= std7



test = test.to_numpy()

test = torch.FloatTensor(test).to(device)



stdY = np.array(stdY)

stdY = torch.FloatTensor(stdY)



meanY = np.array(meanY)

meanY = torch.FloatTensor(meanY)



prediction = model(test) * stdY +meanY
submit = pd.read_csv('/kaggle/input/metro/submit.csv')



for i in range(len(submit)):

    submit['Expected'][i] = prediction[i]



submit