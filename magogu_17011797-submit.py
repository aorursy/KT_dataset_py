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

import torch.optim as optim

import numpy as np

# gpu 연결

device = 'cuda' if torch.cuda.is_available() else 'cpu'





# 데이터 로드

xy_data = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv',usecols=range(1,10)) #raw데이터는 8열이 10번째 열이라 10으로 씀

x_test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv',usecols=range(1,9))



x_data = xy_data.iloc[:, 0:8]

y_data = xy_data.iloc[:,[-1]]

y_data = y_data.squeeze()

print(x_data)

print(y_data)

print(x_data.shape)

print(y_data.shape)

# 데이터 분할- train, validation, test로 나뉨

from sklearn.model_selection import train_test_split



# x,y(numpy) 데이터를 각각 3:7비율로 분리.(stratify=y로 한쪽에 쏠려서 분배되는 것을 방지!)

x_train, x_validation, y_train, y_validation = train_test_split(x_data, y_data, test_size=0.3, random_state=1, stratify = y_data) 

print(x_train.shape)

print(x_validation.shape)



print(y_train.shape)

print(y_validation.shape)
# 입력데이터의 표준화

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(x_data)

x_train_std = sc.transform(x_train) #training data의 표준화

x_validation_std = sc.transform(x_validation)

x_test_std = sc.transform(x_test)
# 모형 추정 혹은 사례중심학습(knn적용)

from sklearn.neighbors import KNeighborsClassifier # knn 불러오기

knn = KNeighborsClassifier(n_neighbors=3, p=2)



knn.fit(x_train_std, y_train) #모델 피팅 과정



y_train_pred = knn.predict(x_train_std) #y값 예측치

y_validation_pred = knn.predict(x_validation_std) #모델을 적용한 validation data의 y값 예측치

print(y_train_pred.shape)

print(y_validation_pred.shape)



#---test data 적용---#

y_test_pred = knn.predict(x_test_std)





print('Misclassified training smaples: %d' %(y_train!=y_train_pred).sum()) # 오분류 데이터 갯수 확인

print('Misclassified test smaples: %d' %(y_validation!=y_validation_pred).sum()) # 오분류 데이터 갯수 확인



from sklearn.metrics import accuracy_score

print(accuracy_score(y_validation, y_validation_pred))
#결과 분석 (Confusion matrix로 확인)

from sklearn.metrics import confusion_matrix #오분류표 작성을 위한 모듈 import

conf = confusion_matrix(y_true = y_validation, y_pred = y_validation_pred)

print(conf) # 행방향(y)- GT값, 열방향(x)- predict
submit = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')



for i in range(len(submit)):

    submit['Label'][i]=y_test_pred[i].item()

submit
submit = submit.astype(np.int32) 

submit.to_csv('submit.csv',index=False,header=True)