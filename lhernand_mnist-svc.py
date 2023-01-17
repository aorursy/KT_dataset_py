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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
train.shape
test.head()
test.shape
train.isnull().sum()
test.isnull().sum()
train.label.value_counts().sort_index().plot(kind='bar')
train.label.value_counts().sort_index()
X = train.drop('label', axis = 1)
y = train.label

X.shape , y.shape
x = X.astype('float32')/255
x.shape
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 0)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
X = test.astype('float32')/255
predicted = svc.predict(X)
predicted
test['index'] = test.index
submmission = pd.DataFrame()
submmission['ImageId'] = test['index']
submmission['Label'] = predicted
submmission.head()
submmission.to_csv('submmission.csv', index='False')