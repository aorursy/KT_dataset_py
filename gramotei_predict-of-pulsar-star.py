# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path = '../input/pulsar_stars.csv'
data = pd.read_csv(path)
data.describe()
data.head()
data.info()
data.replace([0, 1], ['0', '1'], inplace=True)
X = data.drop('target_class', axis=1)
Y = data['target_class']
print(Y.shape)
from sklearn.preprocessing import StandardScaler
scaleCoder = StandardScaler()
scaleCoder.fit_transform(X)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
model = DecisionTreeClassifier()
model.fit(train_x, train_y)
predicted_value = model.predict(test_x)
print(classification_report(test_y, predicted_value))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_x, train_y)
predicted_value = model.predict(test_x)
print(classification_report(test_y, predicted_value))
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train_x, train_y)
predicted_value = model.predict(test_x)
print(classification_report(test_y, predicted_value))