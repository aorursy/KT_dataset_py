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



#데이터 로더

train = pd.read_csv("/kaggle/input/logistic-classification-diabetes-knn/train.csv")

test_data = pd.read_csv("/kaggle/input/logistic-classification-diabetes-knn/test_data.csv")

train.head()
train_x = train.loc[:,'0':'7']

test_x = test_data.loc[:,'0':'7']

train_x.head()
train_x.info()
train_y = train['8']

train_y
#스케일 조정

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(train_x)

X_train_new = sc.transform(train_x)

test_x = sc.transform(test_x)
#데이터분할

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_new, train_y, test_size=0.3, random_state=3, stratify= train_y)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#학습

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='distance', leaf_size=20)

knn.fit(X_train, y_train)
train_pred = knn.predict(X_train)

test_pred = knn.predict(X_test)

print("Misclassified training: %d" %(y_train!=train_pred).sum())

print("Misclassified test: %d" %(y_test!=test_pred).sum())
#결과분석

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, test_pred))
from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_true=y_test, y_pred = test_pred)

print(conf)
#예측

test_pred = knn.predict(test_x)

test_pred
#제출

submission = pd.read_csv("/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv")

submission.head()
for i in range(len(test_pred)):

    submission["Label"][i] = test_pred[i]

submission['Label'] = submission['Label'].astype(int)
submission.head()
submission.to_csv('submission.csv', index=False, header=True)