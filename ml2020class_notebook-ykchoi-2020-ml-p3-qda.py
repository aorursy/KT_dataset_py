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
# 데이터 이상하네

train = pd.read_csv("/kaggle/input/2020-ml-p3/train.csv", index_col=0)

X_test = pd.read_csv("/kaggle/input/2020-ml-p3/test.csv", index_col=0)

submit = pd.read_csv("/kaggle/input/2020-ml-p3/sample_submission.csv",index_col = 0)
print(train.shape)

print(X_test.shape)

print(submit.shape)

# 학습데이터와 라벨 분리

# 데이터 label 로 안들어가 있네? 흠;; 데이터 바꿔야할듯;;



X_train = train.drop('label', axis=1)

y_train = train['label']



y_train.head(100)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



cld = QuadraticDiscriminantAnalysis()

cld.fit(X_train,y_train)

y_train_pred = cld.predict(X_train)

y_test_pred = cld.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, y_train_pred))

submit['label'] = y_test_pred

submit.to_csv("mySubmition.csv")
submit.head(100)