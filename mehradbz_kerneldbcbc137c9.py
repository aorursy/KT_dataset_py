# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import os

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



data = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")

data.sample(10)
X=np.asarray(data.iloc[:,1:-1].dropna(),dtype=np.float32)

print (X.shape)

Y=np.asarray(data.iloc[:,-1])

print (Y, Y.shape, len(np.unique(Y))) # so we have 9 classes...
data.describe()
np.sum(data.iloc[:,1:94] > 40)
data.isnull().sum()
from sklearn.preprocessing import StandardScaler

X_standard = StandardScaler().fit_transform(X)

X_standard
from sklearn.model_selection import cross_validate

from sklearn.cluster import KMeans

km = KMeans(n_clusters=9)

cv_results = cross_validate(knn, X_standard, Y, cv=5)

cv_results['test_score']
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

parameters = {'n_estimators': [50,100,200]}

clf = GridSearchCV(rfc, parameters,cv=5)

clf.fit(X_standard, Y)

cv_results = clf.cv_results_

cv_results
## without standardization

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

parameters = {'n_estimators': [50,100,200]}

clf = GridSearchCV(rfc, parameters,cv=5)

clf.fit(X, Y)

cv_results = clf.cv_results_

cv_results
test_data = pd.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")

test_data.head()

X_test=np.asarray(test_data.iloc[:,1:].dropna(),dtype=np.float32)

#print(X_test.shape)

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X,Y)

y = rfc.predict(X_test)



c1 = np.zeros(X_test.shape[0],dtype=int)

c1[y=='Class_1'] = 1

c2 = np.zeros(X_test.shape[0],dtype=int)

c2[y=='Class_2'] = 1

c3 = np.zeros(X_test.shape[0],dtype=int)

c3[y=='Class_3'] = 1

c4 = np.zeros(X_test.shape[0],dtype=int)

c4[y=='Class_4'] = 1

c5 = np.zeros(X_test.shape[0],dtype=int)

c5[y=='Class_5'] = 1

c6 = np.zeros(X_test.shape[0],dtype=int)

c6[y=='Class_6'] = 1

c7 = np.zeros(X_test.shape[0],dtype=int)

c7[y=='Class_7'] = 1

c8 = np.zeros(X_test.shape[0],dtype=int)

c8[y=='Class_8'] = 1

c9 = np.zeros(X_test.shape[0],dtype=int)

c9[y=='Class_9'] = 1

o = test_data[['id']]

#c1.shape

o['Class_1'] = pd.Series(c1)

o['Class_2'] = pd.Series(c2)

o['Class_3'] = pd.Series(c3)

o['Class_4'] = pd.Series(c4)

o['Class_5'] = pd.Series(c5)

o['Class_6'] = pd.Series(c6)

o['Class_7'] = pd.Series(c7)

o['Class_8'] = pd.Series(c8)

o['Class_9'] = pd.Series(c9)

o.head()

o.to_csv("submit.csv", index=False)