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
x_train = pd.read_csv('../input/data-science-london-scikit-learn/train.csv', header = None)
y_train = pd.read_csv('../input/data-science-london-scikit-learn/trainLabels.csv', header = None)
x_test = pd.read_csv('../input/data-science-london-scikit-learn/test.csv', header = None)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
y_train = y_train.values.ravel()
clf.fit(x_train, y_train)
y_test = clf.predict(x_test)
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
y_train = y_train.values.ravel()
clf.fit(x_train, y_train)
y_test = clf.predict(x_test)
pred = pd.DataFrame(y_test)
pred['Solution'] = y_test
pred['Id'] = [i for i in range(1,y_test.shape[0] + 1)]
pred[['Id','Solution']]
pred.to_csv('Submission.csv', index =False)