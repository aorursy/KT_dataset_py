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
df_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header = None)

df_labels = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header = None)

df_test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header = None)
df_train.head(10)
df_labels.head()
df_test.head()
df_train.columns.values
df_train.shape
df_labels.shape
df_test.shape
df_train.isnull().sum().sum()
df_test.isnull().sum().sum()
from sklearn.model_selection import train_test_split

X = df_train
y = df_labels

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

i = 3
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, np.ravel(y_train))
from sklearn import metrics

prediction = knn.predict(X_test)
print('Test set accuracy: ', metrics.accuracy_score(prediction, y_test))
# Classify the test dataset provided

test = df_test.to_numpy()
prediction = knn.predict(test)

output = pd.DataFrame(columns = ['Id', 'Class'])
output.Id = range(df_test.shape[0])
output.Class = prediction

print(output.head())
print(output.tail())
# Check some more classifiers
from sklearn.tree import DecisionTreeClassifier

tree =  DecisionTreeClassifier()
tree.fit(X_train, y_train)
prediction = tree.predict(X_test)
print('Test set accuracy: ', metrics.accuracy_score(prediction, y_test))
from sklearn.svm import SVC

svm = SVC().fit(X_train, np.ravel(y_train))
prediction = svm.predict(X_test)
print('Test set accuracy: ', metrics.accuracy_score(prediction, y_test))