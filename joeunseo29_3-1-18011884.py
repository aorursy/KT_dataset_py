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

import pandas as pd

import numpy as np

import torch.optim as optim

import torch.nn.functional as F



torch.manual_seed(1)
xy= pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv', header= None, skiprows=1)



X=xy.loc[:, 1:8]  #범위 조심하면서 바꾸기 

y=xy.loc[:,[9]]  #괄호 쳐서 하렴 ,,꼭 [] 기억해 





X= np.array(X)

y=np.array(y)

X= torch.FloatTensor(X)

y=torch.LongTensor(y)



print(y)

print(X)
#테스터할 데이터 가져오기 

test= pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv', header=None, skiprows=1)

x_test= test.loc[:, 1:8]

x_test= np.array(x_test)

x_test= torch.FloatTensor(x_test)

print(x_test)

#knn적용 

from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 5 , p=2)  #5개의 이웃 , 거리 방식

knn.fit(X, y)
y_pred = knn.predict(X)  #X 데이터의 y값 예측치 

from sklearn.metrics import accuracy_score

print(accuracy_score(y, y_pred))
from sklearn.neighbors import KNeighborsClassifier 

y_test_pred = knn.predict(x_test)



y_test_pred=np.array(y_test_pred).reshape(-1,1)

y_test_pred =torch.LongTensor(y_test_pred)



print(y_test_pred)

id= np.array([i for i in range(50)]).reshape(-1,1)



result= np.hstack([id, y_test_pred])  #합치기 



submit= pd.DataFrame(result, columns=["Id",  "Label"])

submit.to_csv("result1.csv", index=False, header=True)


