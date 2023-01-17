# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.keys()
len(data.keys())
data.isnull().sum()
drop_columns = ['Unnamed: 32','id']

data = data.drop(drop_columns, axis=1)
data.shape
data.head()
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

data.head()
data['diagnosis'].value_counts()
labels = ['Benign','Malignant']

classes = pd.value_counts(data['diagnosis'], sort = True)

classes.plot(kind = 'bar', rot=0)

plt.title("Transaction class distribution")

plt.xticks(range(2), labels)

plt.xlabel("Class")

plt.ylabel("Frequency")
y = pd.DataFrame(data['diagnosis'])

X = data.drop(['diagnosis'], axis = 1)



y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report
svc_model = SVC()
svc_model.fit(X_train, y_train)
model_predict = svc_model.predict(X_test)
model_predict
cm = confusion_matrix(y_test, model_predict)



sns.heatmap(cm, annot=True)