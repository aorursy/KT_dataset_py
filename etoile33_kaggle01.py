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
        
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read csv (comma separated value) into data
train = pd.read_csv('../input/data-science-london-scikit-learn/train.csv', header=None)
trainLabel = pd.read_csv('../input/data-science-london-scikit-learn/trainLabels.csv', header=None)
test = pd.read_csv('../input/data-science-london-scikit-learn/test.csv', header=None)

train.describe()
test.describe()
trainLabel.describe()
train.head()
#missing value 
train.isnull().sum().sort_values(ascending=False).head()
#隨機森林模型
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 切分訓練集/測試集
X_train,X_test,y_train,y_test = train_test_split(train,trainLabel,test_size=0.30, random_state=101)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc_randomforest =accuracy_score(y_pred, y_test) * 100
print(acc_randomforest)
