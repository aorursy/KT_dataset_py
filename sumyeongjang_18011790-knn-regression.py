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
import numpy as np

import torch

import pandas as pd



xy_train=np.loadtxt('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(1, 6))

x_train=torch.from_numpy(xy_train[:, 0:-1])

y_train=torch.from_numpy(xy_train[:, [-1]])



x_test=np.loadtxt('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(1, 5))

x_test=torch.from_numpy(x_test)



print(x_train)

print(y_train)

print(x_test)
from sklearn.neighbors import KNeighborsRegressor

regressor=KNeighborsRegressor(n_neighbors=9)
regressor.fit(x_train, y_train)

predict = regressor.predict(x_test)
submission=pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')

for i in range(len(predict)):

  submission["Expected"][i]=int(predict[i])

submission["Expected"]=submission["Expected"].astype(int)

submission.to_csv('my_submission.csv', index=False, header=True)
submission