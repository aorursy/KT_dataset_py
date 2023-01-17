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
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
df=df[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time','DEATH_EVENT']]
df['Old'] = 0

df.loc[df.age >= 80.0 , 'Old'] = 1

df.drop(['age'],axis= 1,inplace=True)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
X = df.drop(['DEATH_EVENT'],axis = 1)

y = df['DEATH_EVENT']

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

svm_pipeline = make_pipeline(StandardScaler(),SVC())

score = cross_val_score(svm_pipeline, X_train, y_train, cv=5)

print("Accuracy: %f " % (score.mean()))
from sklearn.tree import DecisionTreeClassifier



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

DT_pipeline = make_pipeline(StandardScaler(),DecisionTreeClassifier())

scores = cross_val_score(svm_pipeline, X_train, y_train, cv=30)

print("Accuracy: %f " % (scores.mean()))
best = 0;

for i in range(3,21):    

    model = KNeighborsClassifier(n_neighbors = i)

    model.fit(X_train,y_train)

#    print(model.score(X_test,y_test))

    if model.score(X_test,y_test) > best:

        best = model.score(X_test,y_test)

        best_n = i

print("Best Score and N_neigbors:")

print(best,",",best_n)
KNN_mod = KNeighborsClassifier(n_neighbors = best_n)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

KNN_pipeline = make_pipeline(StandardScaler(),DecisionTreeClassifier())

scores = cross_val_score(svm_pipeline, X_train, y_train, cv=30)

print("Accuracy: %f " % (scores.mean()))
from sklearn.linear_model import LogisticRegression

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

KNN_pipeline = make_pipeline(StandardScaler(),LogisticRegression())

scores = cross_val_score(svm_pipeline, X_train, y_train, cv=30)

print("Accuracy: %f " % (scores.mean()))