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


train = pd.read_csv("/kaggle/input/2020-ml-p1/train.csv")

X_test = pd.read_csv("/kaggle/input/2020-ml-p1/test.csv")

submit = pd.read_csv("/kaggle/input/2020-ml-p1/test_sampleSubmission.csv",index_col = 0)
print(train.shape)

print(X_test.shape)

print(submit.shape)
# 데이터 확인 후 전처리



# NAN, missing, 범주형 데이터 확인

pd.options.display.max_columns = None

train.head()

#X_test.head()
# 학습데이터와 라벨 분리

X_train = train.drop('diagnosis', axis=1)

y_train = train['diagnosis']



# NAN 제거

X_train['Unnamed: 32'] = 0

X_test['Unnamed: 32'] = 0
pd.options.display.max_columns = None

X_train.head()
print(X_train.shape)

print(X_test.shape)



# LDA를 이용한 모델 학습 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

cld = LinearDiscriminantAnalysis()

cld.fit(X_train,y_train)







# 학습된 모델을 이용한 추론

y_train_pred = cld.predict(X_train)

y_test_pred = cld.predict(X_test)





from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, y_train_pred))



submit.head(5)
submit['diagnosis'] = y_test_pred
submit.head(20)
submit.to_csv("mySubmition.csv")