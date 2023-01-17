# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline





df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

print(df.shape)

df.head()
df.info()
df.describe()
X=df.loc[:,['CreditScore', 'Age', 'Tenure', 'Balance'

                  , 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values

y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25,random_state =0)
from sklearn.preprocessing import StandardScaler

sc_X =StandardScaler()

X_train =sc_X.fit_transform(X_train)

X_test =sc_X.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.svm import SVC

classifier = SVC(kernel = 'poly', degree = 2, random_state = 0) #degree for non-linear

classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)