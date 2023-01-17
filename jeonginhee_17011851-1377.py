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

import pandas as pd

import torch

import seaborn as sns
train_data=pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

x_train=train_data.loc[:,'avgTemp':'rainFall']

y_train=train_data.loc[:,'avgPrice']

y_train
test_data=pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

x_test=test_data.loc[:,'avgTemp':'rainFall']

x_test
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x_train)

x_train_std=sc.fit_transform(x_train)

x_test_std=sc.fit_transform(x_test)

x_train_std
from sklearn.neighbors import KNeighborsRegressor

knn_regressor=KNeighborsRegressor(leaf_size=10,n_neighbors=150,weights="distance",p=2)

knn_regressor.fit(x_train_std,y_train)
y_train_pred=knn_regressor.predict(x_train_std) #train data의 y값 예측치(표준화된 train_data)

y_test_pred=knn_regressor.predict(x_test_std)  #모델을 적용한 test data의 y값 예측치(표준화된 test_data)

print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum()) #오분류 데이터 갯수 확인

#print('Misclassified test samples: %d' %(y_test!=y_test_pred).sum()) #오분류 데이터 갯수 확인

print(y_train_pred)

submit=pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')

for i in range(len(y_test_pred)):

  submit['Expected']=y_test_pred

submit=submit.astype(int)

submit.to_csv('submit.csv',mode='w',header=True,index=False)

submit