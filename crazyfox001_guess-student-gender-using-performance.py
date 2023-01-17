# Let's Try If we can Guess Student gender By Their Perfomance
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
Data = pd.read_csv("../input/StudentsPerformance.csv")

Data.head()
X = Data.drop("gender",axis=1)

y = Data.iloc[:,0:1]

dummy_X = pd.get_dummies(X,drop_first=True)

dummy_X.head()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

y = LE.fit_transform(y)

y[0:10]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(dummy_X)

transformed_X = scaler.transform(dummy_X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.33, random_state=42)
from sklearn import svm

clf = svm.SVC(gamma='scale')

clf.fit(X_train, np.ravel(y_train,order='C')) 
predicted_y = clf.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predicted_y)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicted_y)
# Have Ok Accuracy Let's see F1 Score
from sklearn.metrics import f1_score

f1_score(y_test, predicted_y, average='macro')