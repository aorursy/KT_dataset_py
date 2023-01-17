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

import numpy as np

import pandas as pd

import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

device = torch.device("cuda")

sc = StandardScaler()

torch.manual_seed(1)



# Load data

xy_data = pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

x_test = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

submit = pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')
# Preprocessing

# Train data

xy_data = np.array(xy_data)

x_train = torch.FloatTensor(xy_data[:,:-1])

y_train = torch.FloatTensor(xy_data[:,-1])



# Test data 

x_test = np.array(x_test)

x_test = torch.FloatTensor(x_test)

for i in range(len(x_train)):

  x_train[i][0]%=10000.0

  x_train[i][0]/=100.0

for i in range(len(x_test)):

  x_test[i][0]%=10000.0

  x_test[i][0]/=100.0
for i in range(len(xy_data)):

  xy_data[i][0]/=10000

  xy_data[i][0]%=100

x_train_add= xy_data[:,0].reshape(-1,1)



x_test_addd = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

x_test_addd = np.array(x_test_addd)

for i in range(len(x_test_addd)):

  x_test_addd[i][0]/=10000

  x_test_addd[i][0]%=100

x_test_add= x_test_addd[:,0].reshape(-1,1)
x_train =np.array(x_train)

x_test =np.array(x_test)
# x_train = np.hstack([x_train_add,x_train])

# x_test = np.hstack([x_test_add,x_test])
sc.fit(x_train)

x_train_std = sc.transform(x_train)

x_test_std = sc.transform(x_test)

#x_train_std = torch.FloatTensor(x_train_std).to(device)

#x_test_std = torch.FloatTensor(x_test_std).to(device)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=25, p = 2)

knn.fit(x_train_std,y_train) # knn 학습
y_test_pred = knn.predict(x_test_std)
for i in range(len(submit)):

  submit['Expected'][i] = y_test_pred[i];
submit.to_csv('submit.csv', mode='w', header=True, index= False)
#!kaggle competitions submit -c mlregression-cabbage-price -f 'submit.csv' -m 'hjkim_submission'