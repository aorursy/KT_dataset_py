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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv('../input/titanic/train.csv')

train.head()
train.isnull().sum()
train.corr()
test=pd.read_csv('../input/titanic/test.csv')

test.head()
test.isnull().sum()
test.corr()
gender=pd.read_csv("../input/titanic/gender_submission.csv")

gender.head()
train.drop('Cabin',axis=1,inplace=True)

train.head()
train['members']=train['Parch']+train['SibSp']

train.head()
test.drop('Cabin',axis=1,inplace=True)

test.head()
test['members']=test['Parch']+test['SibSp']

test.head()
grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=10)

grid.add_legend();
grid = sns.FacetGrid(train, row='Embarked', height=2.2, aspect=1.6)

grid.map(sns.boxplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
X_train=train.iloc[:,[0,2,3,4,5,6,7,8,9,10,11]].values

X_train
y_train=train.iloc[:,1].values

y_train
X_test=test.iloc[:,:].values

X_test
y_test=gender.iloc[:,1].values

y_test
from sklearn.impute import SimpleImputer

imp=SimpleImputer(missing_values=np.NaN,strategy='mean')

X_train[:,4:5]=imp.fit_transform(X_train[:,4:5])

X_test[:,4:5]=imp.transform(X_test[:,4:5])

print(X_train)

print(X_test)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])

X_test[:, 3] = labelencoder_X.fit_transform(X_test[:, 3])
X_tr=X_train[:,[1,3,4,10]]

X_te=X_test[:,[1,3,4,10]]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_tr = sc.fit_transform(X_tr)

X_te = sc.transform(X_te)
from sklearn.metrics import make_scorer, accuracy_score
#RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

classifier1 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

classifier1.fit(X_tr, y_train)
y_pred1 = classifier1.predict(X_te)

y_pred1
#SCORE

score=round(accuracy_score(y_test,y_pred1)*100,2)

score
#SUPPORT VECTOR CLASSIFIER

from sklearn.svm import SVC

classifier2 = SVC(kernel = 'rbf', random_state = 0)

classifier2.fit(X_tr, y_train)
y_pred2 = classifier2.predict(X_te)

y_pred2
#SCORE

score=round(accuracy_score(y_test,y_pred2)*100,2)

score
#NAIVE BAYES CLASSIFIER

from sklearn.naive_bayes import GaussianNB

classifier3 = GaussianNB()

classifier3.fit(X_tr, y_train)

y_pred3 = classifier3.predict(X_te)

y_pred3
#SCORE

score=round(accuracy_score(y_test,y_pred3)*100,2)

score
#KNN CLASSIFIER

from sklearn.neighbors import KNeighborsClassifier

classifier4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier4.fit(X_tr, y_train)
y_pred4 = classifier4.predict(X_te)

y_pred4
#SCORE

score=round(accuracy_score(y_test,y_pred4)*100,2)

score
print("Original Values")

print(y_test)

print("Predicted Values")

print(y_pred2)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred2)

cm
my_pred=pd.DataFrame({'PassengerId': X_test[:,0], 'Survived': y_pred2})

sns.catplot(x ="Survived", kind ="count", data = my_pred) 
my_pred.to_csv("MySubmission",index=False)

print("successful")