# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
data.head(5)
data.info()
data.describe()
data.shape
from sklearn.ensemble import ExtraTreesClassifier
X = data.iloc[:, 1:5].values
y = data.iloc[:, 5].values
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
X_test
clf =  ExtraTreesClassifier()
clf.fit(X_train, y_train)
clf._feature_importances_