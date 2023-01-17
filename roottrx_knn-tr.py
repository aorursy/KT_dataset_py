# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import sys

# Hataları almamak için
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.head()
y = data.iloc[:, -1:].values
x = data.iloc[:, 0:-1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42 ) 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric = "minkowski")
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
print(cm)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
print(accuracy)
score_list = []
for i in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))
plt.plot(range(1, 15), score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
