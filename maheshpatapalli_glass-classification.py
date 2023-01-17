# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/glass/glass.csv')
dataset.info()
dataset.head()
dataset.isnull().sum()
dataset['Type'].value_counts()
x = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 177013)
for i in range(x_train.shape[0]):

    print('x_train {0} == y train {1}'.format(x_train[i][0],y_train[i]))
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(criterion = 'entropy',splitter = 'best')

DT.fit(x_train,y_train)
from sklearn.ensemble import RandomForestClassifier

RT = RandomForestClassifier(n_estimators = 500,random_state = 177013)

RT.fit(x_train,y_train)
from sklearn.svm import SVC

SV = SVC(kernel = 'rbf')

SV.fit(x_train,y_train)
from xgboost import XGBClassifier

xboost = XGBClassifier(n_estimators = 700, learning_rate = 0.1)

xboost.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred_X =xboost.predict(x_test)

print(y_pred_X)

cm_X = confusion_matrix(y_test,y_pred_X)

print('Accuracy == {0}'.format(accuracy_score(y_test,y_pred_X)))

print(cm_X)
y_pred_S = SV.predict(x_test)

print(y_pred_S)

cm_s = confusion_matrix(y_test, y_pred_S)

print('Accuracy == {0}'.format(accuracy_score(y_test,y_pred_S)))

print(cm_s)
y_pred_R = RT.predict(x_test)

print(y_pred_R)

cm_r = confusion_matrix(y_test,y_pred_R)

print("accuracy == {0}".format(accuracy_score(y_test,y_pred_R)))

print(cm_r)
y_pred = DT.predict(x_test)

print(y_pred)

cm = confusion_matrix(y_test,y_pred)

print("accuracy == {0}".format(accuracy_score(y_test,y_pred)))

print(cm)