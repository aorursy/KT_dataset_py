# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')

sample_submission = pd.read_csv('../input/learn-together/sample_submission.csv')

sample_submission.head()
X = train.drop(['Cover_Type'], axis=1)

y = train['Cover_Type']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
from sklearn import metrics
from sklearn.svm import SVC

classifier = SVC()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_pred = y_pred.tolist()

result = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})

result
print(classifier.score(X_test, y_test))
# linear_svc = SVC(kernel='linear').fit(X_train, y_train)
#rbf_svc = SVC(kernel='rbf').fit(X_train, y_train)
# print('linear¥t: ', linear_svc.score(X_test, y_test))
#print('rbf¥t: ', rbf_svc.score(X_test, y_test))
# c_range = [0.01, 0.05, 0.1, 0.5, 1, 10, 100]

# gamma_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]



# for c in c_range:

#     for g in gamma_range:

#         accuracy = SVC(kernel='rbf', gamma=g, C=c).fit(X_train, y_train).score(X_test, y_test)

#         print('C={0:7.3f}¥t, gamma={1:7.3f}¥t, accuracy={2:.3f}'.format(c, g, accuracy))
# 精度を計算するmetricsライブラリをインポートします

from sklearn import metrics



# # テストデータを予測します

y_pred = classifier.predict(X_test)



# # 精度を計算します。算出には metrics.accuracy_scoreを使用します。

print(metrics.accuracy_score(y_test, y_pred))
test_pred = classifier.predict(test)

sample_submission['Cover_Type'] = test_pred

sample_submission.to_csv('submission.csv', index=False)
!ls