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

import numpy

import torch

import torch.optim as optim



xy_data = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv') #year  avgTemp  minTemp  maxTemp  rainFall  avgPrice

x_test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')



x_data = xy_data.iloc[:,0:5] # 6전까지 

y_data = xy_data.iloc[:,-1]

print(x_data.shape, y_data.shape)

print(y_data)
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 3, weights = "distance") #가중회귀



regressor.fit(x_data, y_data) #학습해라



#---test data 적용---#

y_test_pred = regressor.predict(x_test) #예측해라

submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')



for i in range(len(submit)):

    submit['Expected'][i]=y_test_pred[i].item()

print(submit)
submit.to_csv('submit.csv',index=False,header=True)