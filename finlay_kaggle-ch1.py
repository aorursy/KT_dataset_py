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
!ls ../input/
for idx in range(10):
    if idx % 3:
        print(idx)
arr = list(range(10))
df = pd.Series(arr)
df
arr = {x: x*x for x in range(10)}
df = pd.Series(arr)
df
dates = pd.date_range('today',periods=6) # 定义时间序列作为 index
num_arr = np.random.randn(6,4) # 传入 numpy 随机数组
columns = ['A','B','C','D'] # 将列表作为列名
df = pd.DataFrame(num_arr, index = dates, columns = columns)
df
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df
df.info()
df.describe()
df.iloc[:10]
df['MSSubClass']
df[['MSSubClass', 'MSZoning']]
df[df['MSZoning'] == 'RL']
df[(df['MSZoning'] == 'RL') & (df['MSSubClass'] > 60)]
df['MSSubClass'].mean()
df.groupby(['MSZoning'])['MSSubClass'].mean()
df['MSZoning'].value_counts()
df['MSZoning'].map({'RL':1, 'RM': 2, 'FV':3, 'RH': 4})
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.datasets import make_classification
data_set = make_classification(n_samples=1000, n_features=50)
train_set, test_set, train_label, test_label = train_test_split(data_set[0], data_set[1], test_size=0.2)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_set, train_label)
accuracy_score(clf.predict(test_set), test_label), roc_auc_score(clf.predict(test_set), test_label)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(train_set, train_label)
accuracy_score(clf.predict(test_set), test_label), roc_auc_score(clf.predict(test_set), test_label)
clf = LogisticRegression(C=1)
clf.fit(train_set, train_label)
accuracy_score(clf.predict(test_set), test_label), roc_auc_score(clf.predict(test_set), test_label)
clf = LogisticRegression(C=10)
clf.fit(train_set, train_label)
accuracy_score(clf.predict(test_set), test_label), roc_auc_score(clf.predict(test_set), test_label)
