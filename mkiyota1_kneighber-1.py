# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')  ## Load the Training Data set

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.head()
from sklearn.neighbors import KNeighborsClassifier
X = train.drop(['Cover_Type'], axis=1)

y = train['Cover_Type']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# 精度を計算するmetricsライブラリをインポートします

from sklearn import metrics
# kを1から100まで値を変化させます

k_range = range(1, 100)



accuracy = []

accuracy_train = []



# kを変化させて、精度を計算します。kの数だけ繰り返します

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    accuracy.append(knn.score(X_test, y_test))
# k_rangeとaccuracyをプロットします

plt.plot(k_range, accuracy)



# 【時間があれば】x軸のラベルを K for kNNとします

plt.xlabel('k for kNN')



# 【時間があれば】y軸のラベルを Testing Accuracyとします

plt.ylabel('Testing Accuracy')



# 描画します

plt.show()
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X, y)

test_pred = knn.predict(test)
sample_submission['Cover_Type'] = test_pred

sample_submission.to_csv('submission.csv', index=False)
!ls