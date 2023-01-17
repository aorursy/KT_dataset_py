# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/diabetes-dataset/diabetes2.csv")

df.columns
missing_cols = [col for col in df.columns if df[col].isnull().any()]

missing_cols
X = pd.DataFrame(df.iloc[:,0:8])

y = pd.DataFrame(df.iloc[:,8])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LogisticRegression(max_iter=250)

clf.fit(X_train, y_train['Outcome'])
y_test_hat = clf.predict(X_test)

print(classification_report(y_test,y_test_hat))
plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Diabetic', 'Non-diabetic'],)
y_train_hat = clf.predict(X_train)

print(classification_report(y_train,y_train_hat))
plot_confusion_matrix(clf, X_train, y_train, cmap=plt.cm.Blues, display_labels=['Diabetic', 'Non-diabetic'],)