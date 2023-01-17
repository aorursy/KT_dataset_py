#  Lets Try to Guess Students Ethinicity Using Their Prefomance In Exams.

#  Thanx God They Didn't used Real Race Name. Otherwise No matter the result. I'll be known as Racist

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
X = Data.drop("race/ethnicity",axis=1)

y = Data.iloc[:,1:2]

y.head()

X.head()
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
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicted_y)

## You see this is Terrible Model Let's Try Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

clf.fit(X_train, y_train)
r_predicted_y = clf.predict(X_test)
accuracy_score(y_test, r_predicted_y)
print(r_predicted_y)
# So It's seems like Race Dosen't have anything to do with Your Grades ( Or We need Bigger Database)