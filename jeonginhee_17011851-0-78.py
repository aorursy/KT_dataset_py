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
train_data=pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')

test_data=pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')

print(train_data)
type(train_data)
x_train=train_data.loc[:,'0':'7']

y_train=train_data.loc[:,'8']

x_test=test_data.loc[:,'0':'7']

x_test
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x_train)

x_train_std=sc.transform(x_train)

x_test_std=sc.transform(x_test)

x_train_std
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3,p=1)

knn.fit(x_train_std,y_train)
y_train_pred=knn.predict(x_train_std) #train data의 y값 예측치(표준화된 train_data)

y_test_pred=knn.predict(x_test_std)  #모델을 적용한 test data의 y값 예측치(표준화된 test_data)

print('Misclassified training samples: %d' %(y_train!=y_train_pred).sum()) #오분류 데이터 갯수 확인

#print('Misclassified test samples: %d' %(y_test!=y_test_pred).sum()) #오분류 데이터 갯수 확인
from sklearn.metrics import accuracy_score    #정확도 계산을 위한 모듈 import

print(accuracy_score(y_train,y_train_pred)) # 45개 test sample중 42개가 정확하게 분류됨.
result=pd.read_csv("../input/logistic-classification-diabetes-knn/submission_form.csv")

for i in range(len(y_test_pred)):

  result['Label']=y_test_pred

result=result.astype(int)

print(type(result))
result.to_csv('result.csv', mode='w', header= True, index= False)

result