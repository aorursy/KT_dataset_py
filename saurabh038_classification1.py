# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df = df.drop(['Unnamed: 32'], axis=1)
print(df.isna().sum())
df.columns
x = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']
# print(x.head, y.head)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
bayes_model = GaussianNB()
bayes_model.fit(x_train, y_train)
y_pred = bayes_model.predict(x_test)
print("The accuracy score is "+str(accuracy_score(y_test, y_pred)*100))
plot_confusion_matrix(bayes_model, x_test, y_test)
plt.show()
print(classification_report(y_test, y_pred, labels=["M", "B"]))
import random
actual = list(y_test)
n = 10
a = random.randint(0, len(actual)-n)
temp = list(bayes_model.predict(x_test))
for i in range(a, a+n):
    print("Actual diagnosis: "+actual[i])
    print("Predicted diagnosis: "+ temp[i])
    print("===============================")



tree_model = DecisionTreeClassifier()
tree_model.fit(x_train, y_train)
y_pred = tree_model.predict(x_test)
print("The accuracy score is "+str(accuracy_score(y_test, y_pred)*100))
plot_confusion_matrix(tree_model, x_test, y_test)
plt.show()
print(classification_report(y_test, y_pred, labels=["M", "B"]))
import random
actual = list(y_test)
n = 5
a = random.randint(0, len(actual)-n)
temp = list(bayes_model.predict(x_test))
for i in range(a, a+n):
    print("Actual diagnosis: "+actual[i])
    print("Predicted diagnosis: "+ temp[i])
    print("===============================")