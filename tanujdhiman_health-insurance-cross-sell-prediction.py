# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
train_data = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
train_data.head(5)
test_data = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
test_data.head(5)
train_data.info()
train_data.Vehicle_Age.unique()
train_data.Vehicle_Damage.unique()
train_data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.corr(), annot = True, cmap = 'coolwarm', center = 0)
plt.show()
train_data.shape
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train_data['Gender'] = label.fit_transform(train_data['Gender'])
train_data['Vehicle_Damage'] = label.fit_transform(train_data['Vehicle_Damage'])
train_data.info()
train_data.head(5)
from sklearn.ensemble import ExtraTreesClassifier
train_data =  train_data.drop(['Vehicle_Age'], axis=1)
train_data.head()
train_data.info()
X = train_data.iloc[:, 0:10].values
y = train_data.iloc[:, 10].values
clf =  ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
importances = clf.feature_importances_
importances
max(importances)
X_new = train_data.iloc[:, 0].values
X_new.shape
y.shape
##Import

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
model = DecisionTreeClassifier(random_state=0)
model.fit(X_new.reshape(1, -1), y.reshape(1, -1))
y_pred = model.predict(X_new.reshape(1, -1))
y_pred