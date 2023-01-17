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

torch.manual_seed(1)
#트레인 데이터를 불러오기 

xy= pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv', header= None, skiprows=1)

x_data= xy.loc[:, 1:4]

y_data=xy[5]



x_data=np.array(x_data, dtype=float)

y_data= np.array(y_data, dtype=float)



y_data= y_data.reshape(-1,1)

x_data=torch.FloatTensor(x_data)

y_data=torch.LongTensor(y_data)



print(x_data)

print(y_data)



#테스트 데이터 불러오기 

x_testd=pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv', header=None, skiprows=1)

x_test=x_testd.loc[:, 1:4]

x_test= np.array(x_test)

x_test= torch.FloatTensor(x_test)

print(x_test)
#스케일러 쓰기 

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

sc.fit(x_data)





x_data= sc.transform(x_data)

x_test = sc.transform(x_test)







print(x_data)



print(x_test)

#모델 학습하기 

from sklearn.neighbors import KNeighborsRegressor #회귀 문제 푸는방법

regressor = KNeighborsRegressor(n_neighbors = 1000)

regressor.fit(x_data, y_data)



guesses = regressor.predict(x_test)

guesses


submit= pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')



for i in range(len(guesses)):

  submit['Expected'][i]=guesses[i].item()



print(submit)
submit.to_csv('re.csv', index=False, mode='w')