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
train = pd.read_csv("/kaggle/input/2020-ml-p5/train.csv", index_col=0)

X_test = pd.read_csv("/kaggle/input/2020-ml-p5/test.csv", index_col=0)

submit = pd.read_csv("/kaggle/input/2020-ml-p5/sample_submission.csv",index_col = 0)
print(train.shape)

print(X_test.shape)

print(submit.shape)
# 학습데이터와 라벨 분리

# target이 라벨이라고 알려줘야 할듯 함

X_train = train.drop('target', axis=1)

y_train = train['target']

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



cld = QuadraticDiscriminantAnalysis()

cld.fit(X_train,y_train)

y_train_pred = cld.predict(X_train)

y_test_pred = cld.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, y_train_pred))

submit.head(5)
# sample submittion에 voted인지 label인지 확인 할 것

submit['target'] = y_test_pred

submit.to_csv("mySubmition.csv")