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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
input_file_path = "../input/IRIS.csv"

data = pd.read_csv(input_file_path)

data.head()
data.isnull().sum()
data.describe()
data.groupby('species').size()
data.boxplot(grid = False)
data.hist(grid=False,bins=40,figsize=[12,8])
sns.pairplot(data,kind='scatter')
data.columns
X = data.drop(['species'],axis=1)

y = data['species']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

predictions = knn.predict(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



print(accuracy_score(y_test,predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))