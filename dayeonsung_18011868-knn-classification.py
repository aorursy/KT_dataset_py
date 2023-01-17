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

import numpy as np

train = pd.read_csv("/kaggle/input/logistic-classification-diabetes-knn/train.csv")



X_train = train.loc[:, '0':'7']

y_train = train['8']
X_train = np.array(X_train)

y_train = np.array(y_train)
# KNN 의 적용

from sklearn.neighbors import KNeighborsClassifier # KNN 불러오기

knn=KNeighborsClassifier(n_neighbors=20,p=2) # 5개의 인접한 이웃, 거리측정기준:유클리드

# knn.fit(X_train_std,y_train) # 모델 fitting 과정

knn.fit(X_train,y_train) # 모델 fitting 과정
# y_train_pred=knn.predict(X_train_std) # train data의 y값 예측치

y_train_pred=knn.predict(X_train) # train data의 y값 예측치

# y_test_pred=knn.predict(X_test_std) # 모델을 적용한 test data의 y값 예측치

print('Misclassified training samples: %d' % (y_train!=y_train_pred).sum()) # 오분류 데이터 갯수 확인
test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')

X_test = test.loc[:, '0':'7']

X_test = np.array(X_test)

y_test_pred = knn.predict(X_test)

y_test_pred
submit = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')

for i in range (50):

  submit['Label'][i] = y_test_pred[i]

submit['Label'] =submit['Label'].astype(int)

submit.to_csv("submission.csv", index = False, header = True)