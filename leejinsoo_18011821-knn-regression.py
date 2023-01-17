# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) zwill list all files under the input directory



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

device = torch.device("cpu")

#데이터

xy_data = pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

x_test = pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

submit = pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')
xy_data
xy_np = xy_data.to_numpy()
xy_np[:,0] %= 10000
x_train = xy_np[:,0:-1]

y_train = xy_np[:,-1]
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 3, weights = 'distance')
regressor.fit(x_train,y_train)
x_test
x_test_np = x_test.to_numpy()
x_test_np[:,0]%=10000
x_test_np.shape
x_train.shape
predict = regressor.predict(x_test_np)
predict
submit
for i in range(len(predict)):

    submit['Expected'][i]=predict[i]

submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)
submit