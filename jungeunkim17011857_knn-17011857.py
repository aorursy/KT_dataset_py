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



xy=np.loadtxt('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv',delimiter=',',dtype=np.float32, skiprows=1, usecols=range(1,6))



x_data= torch.from_numpy(xy[:,0:-1])

y_data=torch.from_numpy(xy[:,[-1]])

x_train=torch.FloatTensor(x_data)

y_train=torch.FloatTensor(y_data)

print(x_train.shape, y_train.shape)

print(x_train)
test = np.loadtxt('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv',delimiter=',',dtype=np.float32 , skiprows=1 ,usecols=range(1,5))



x_data = test



x_data=torch.from_numpy(x_data)

x_test= torch.FloatTensor(x_data)

print(x_test.shape)
from sklearn.neighbors import KNeighborsRegressor

regressor= KNeighborsRegressor(n_neighbors =15, weights = "distance")



regressor.fit(x_train,y_train)



guesses=regressor.predict(x_test)

guesses=torch.FloatTensor(guesses).int()
import pandas as pd



predict=guesses.numpy().reshape(-1,1)



id= np.array([i for i in range(731)]).reshape(-1,1)

result=np.hstack([id, predict])

df=pd.DataFrame(result,columns=["Id","Expected"])

df.to_csv("result.csv",index=False,header=True)