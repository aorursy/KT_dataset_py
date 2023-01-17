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
import seaborn as sns

import pandas as pd

import csv

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier



test_data=pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')

train_data=pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')

submission=pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')



submission_form=submission.loc[:,"ID":"Label"]

#print(submission_form)

#print('train_data=', train_data)





#### 8열 칼럼 지우기



test_data=test_data.drop('8', axis=1)

test_data=test_data.loc[:,"0":"7"]

#print('test_data=',test_data)



#크기 확인

#print('train_data.shape= ', train_data.shape)

#print('test_data.shape=', test_data.shape)



#train 데이터 전처리

X_train=pd.DataFrame(train_data)

X_train=X_train.loc[:,"0":"7"]

Y_train=pd.DataFrame(train_data)

Y_train=Y_train.loc[:,"8"]





#X_train=train_data.loc[0:].drop('8', axis=1)

#Y_train=train_data['8']



#print('X_train=', X_train)

#print('Y_train=', Y_train)

#print('X_train.shape=', X_train.shape)

#print('Y_train.shape=', Y_train.shape)



#### KNN 적용하기



knn=KNeighborsClassifier(n_neighbors=3, p=2) #K는 5개, 거리 측정기준은 유클리드

knn.fit(X_train,Y_train)





#### train data의 Y값 예측치



y_train_pred=knn.predict(X_train)

y_test_pred=knn.predict(test_data)

#print('MisClassified training sample: %d' %(Y_train!=y_train_pred).sum())





#print(y_test_pred)

ans=y_test_pred

ans=list(map(int,ans))

#print(ans)

submission['Label']=ans

#print(submission)
submission.to_csv('./submission_form.csv' , columns=["ID","Label"],index = False)


