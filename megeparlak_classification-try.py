# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/heart-disease-uci/heart.csv")

data.head()
data.info()
plt.figure(figsize=(10,10))

import seaborn as sns

sns.countplot(x=data.target)
x_data = data.drop(["target"],axis=1)

y= data.target



x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr = lr.fit(x_train,y_train)

print("Test Accuracy: ",lr.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 4)

knn.fit(x_train,y_train)

print("Test Accuracy: ",knn.score(x_test,y_test))
from sklearn.svm import SVC

svm= SVC(random_state=42)

svm.fit(x_train,y_train)

print("Test Accuracy: ",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Test Accuracy: ",nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=42)

dtc.fit(x_train,y_train)

print("Test Accuracy: ",dtc.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100,random_state=42)

rfc.fit(x_train,y_train)

print("Test Accuracy: ",rfc.score(x_test,y_test))
from sklearn.metrics import confusion_matrix

x_test = rfc.predict(x_test)

cm = confusion_matrix(y_test,x_test)

f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cm,annot=True)

plt.show()