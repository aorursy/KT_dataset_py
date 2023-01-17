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



train = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')



X_train = train.loc[:,'year':'rainFall']

y_train = train['avgPrice']
# KNN 의 적용

from sklearn.neighbors import KNeighborsRegressor # KNN 불러오기

regressor=KNeighborsRegressor(n_neighbors=3,weights="distance") # distance => 가중평균 / default="uniform"



regressor.fit(X_train,y_train) # 모델 fitting 과정
y_train_pred=regressor.predict(X_train) # train data의 y값 예측치



print('Misregressed training samples: %d' % (y_train!=y_train_pred).sum()) # 오분류 데이터 갯수 확인
test = pd.read_csv("/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv")



X_test = test.loc[:,'year':'rainFall']



y_test_pred = regressor.predict(X_test)

y_test_pred
submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')

for i in range (len(y_test_pred)):

  submit['Expected'][i] = y_test_pred[i]

submit['Expected'] =submit['Expected'].astype(int)

submit.to_csv("submission.csv", index = False, header = True)