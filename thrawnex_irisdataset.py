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
# read the data in
data = pd.read_csv("../input/Iris.csv")
data.head()
data.info()
data.isnull().sum()
data.hist()

# making a plot grouped by the class to show the seperability of the classes via their features.

import seaborn as sns
groupedData = data.groupby('Species')
meanValues  = groupedData[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].mean()
print(meanValues.to_string())
 
sns.set(style="ticks")
sns.pairplot(data, hue="Species")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# encode the Species row to numerical values 0,1,2 
data['encoded']  = le.fit_transform(data['Species'])
data.head()
# selected / seperate train and test data.

from sklearn.model_selection import train_test_split
y = data['encoded']
X = data.drop(['Species', 'encoded', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)
# use K Nearest Neighbor for the Classification

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) 
from sklearn.metrics import *

prob01 = knn.predict(X_test)


cm = confusion_matrix(y_test.T.tolist(), prob01)
print("\n" + "Confusion Matrix" + "\n")
print(cm)

score01 = precision_score(y_test.T.tolist(), prob01, average=None)


print("Precision Score Klasse: ")
print (score01)
# use random forest as a classifier.

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
clf.fit(X_train, y_train) 
from sklearn.metrics import *

prob01 = knn.predict(X_test)


cm = confusion_matrix(y_test.T.tolist(), prob01)
print("\n" + "Confusion Matrix" + "\n")
print(cm)

score01 = precision_score(y_test.T.tolist(), prob01, average=None)
acc = accuracy_score(y_test.T.tolist(), prob01)


print("Precision Score Klasse: ")
print (score01)

print("Precision Score Klasse: ")
print (acc)