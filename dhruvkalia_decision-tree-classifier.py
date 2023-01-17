# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import classification_report, plot_confusion_matrix

from sklearn.preprocessing import LabelEncoder

import seaborn as sbn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
missing_values_cols = [col for col in df.columns if df[col].isnull().any()]

missing_values_cols
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

label_encoder = LabelEncoder()

labelled_df = df.copy()

for col in categorical_cols:

    labelled_df[col] = label_encoder.fit_transform(df[col])

labelled_df.head()
correlation=labelled_df.corr()

plt.figure(figsize=(15,15))

sbn.heatmap(correlation,annot=True,cmap=plt.cm.Blues)
labelled_df.drop('veil-type', axis=1, inplace=True)
y = labelled_df.iloc[:, 0]

X = labelled_df.iloc[:, 1:22]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()

parameters = [{'max_depth': [1,2,3,4,5,6,7,8,9,10]}]

clf = GridSearchCV(model, parameters, cv=5, scoring="accuracy")

clf.fit(X_train, y_train)

print(clf.best_params_)
clf = DecisionTreeClassifier(max_depth=9)

clf.fit(X_train, y_train)



y_test_hat = clf.predict(X_test)

print(classification_report(y_test, y_test_hat))

print(plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues,

                            display_labels=['Poison', 'No poison']))
plt.figure(figsize=[20, 10])

tree.plot_tree(clf, rounded= True, filled= True)