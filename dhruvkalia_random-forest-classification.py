# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sbn

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, plot_confusion_matrix

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/glass/glass.csv")

df.describe()
X = df.iloc[:, 0:9]

y = df.iloc[:, 9]
correlation=df.corr()

plt.figure(figsize=(10,10))

sbn.heatmap(correlation,annot=True,cmap=plt.cm.Blues)
model = RandomForestClassifier()

# Applying GridSearchCV to find the best hyperparameters for doing random forest classification

parameters = [{'n_estimators': [10, 20, 50]}]

clf = GridSearchCV(model, parameters, cv=5, scoring="accuracy")

clf.fit(X, y)   

print(clf.best_params_)
clf = RandomForestClassifier(n_estimators=50)

clf.fit(X, y)
y_hat = clf.predict(X)

print(classification_report(y, y_hat))

print(plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues,

                            display_labels=(y.unique())))