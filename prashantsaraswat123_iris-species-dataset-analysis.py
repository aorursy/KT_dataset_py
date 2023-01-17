# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
df = pd.read_csv('../input/Iris.csv')
df.head(10)
df.info()
df.columns
df.describe(include='all')
df.drop('Id', inplace=True, axis=1)
df.head()
df.isnull().any()
df['Species'].value_counts()
sns.pairplot(df, hue='Species', palette='Set2');
sns.distplot(df['SepalLengthCm'], color='red');
sns.boxplot(df['SepalLengthCm'])
label = LabelEncoder()
df['Species'] = label.fit_transform(df['Species'])
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.drop('Species', axis=1))
X = scaled_df
y = df['Species'].as_matrix()
df.head(12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
clf_lr = LogisticRegression(C=10)
clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
print(accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
linear_svm = LinearSVC()
linear_svm.fit(X_train, y_train)
y_pred = linear_svm.predict(X_test)
print(accuracy_score(y_test, y_pred))
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))
