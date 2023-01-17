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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

cancer
print(cancer.data)
print(cancer.target)
print(cancer.target_names)
print(cancer.feature_names)
type(cancer['feature_names']) 
cancer['data'].shape
df_cancer = pd.DataFrame(np.c_[cancer.data, cancer.target], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()
df_cancer.info()
df_cancer.describe().T
plt.figure(figsize=(20,15))

sns.heatmap(df_cancer.corr(), annot=True)
sns.pairplot(df_cancer, hue='target', vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'

 ,'mean smoothness', 'mean compactness', 'mean concavity'])
sns.countplot(df_cancer.target)
sns.scatterplot(x='mean radius', y= 'mean texture', hue='target', data=df_cancer)
x = df_cancer.drop('target', axis=1)

y = df_cancer.target
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
svc_model = SVC()

svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)

X_test_sc = sc.transform(X_test)
svc_model = SVC()

svc_model.fit(X_train_sc, y_train)
y_pred = svc_model.predict(X_test_sc)
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(svc_model, param_grid, cv=5)

grid.fit(X_train_sc, y_train)
grid.best_estimator_
y_pred = grid.predict(X_test_sc)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))