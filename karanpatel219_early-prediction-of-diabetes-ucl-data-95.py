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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('../input/early-stage-diabetesucl/diabetes_data_upload.csv')
data.head()
data['class']=data['class'].replace({'Positive':1,'Negative':0})
data
data.isnull().sum()
data.iloc[:,1:15]
for cat_data in data.iloc[:,1:15]:
    print(cat_data)
from sklearn.preprocessing import LabelEncoder
for cat_data in data.iloc[:,1:16]:
            i=0
            l='l'+str(i)
            l=LabelEncoder()
            data[cat_data]=l.fit_transform(data[cat_data])
            i+=1
data
a=[]
a.append(d[1])
a.append(d[0])
plt.bar([1,0],a)
X=data.drop('class',axis=1)
Y=data['class']

from sklearn.preprocessing import StandardScaler
s=StandardScaler()
X=s.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
clf=[RandomForestClassifier(),AdaBoostClassifier(),LogisticRegression(),SVC(),DecisionTreeClassifier(),KNeighborsClassifier()]
sc=[]
sc_test=[]
for cl in clf:
    model=cl
    model.fit(X_train,Y_train)
    sc.append(model.score(X_train,Y_train))
    sc_test.append(model.score(X_test,Y_test))
sc
sc_test
modeldc=DecisionTreeClassifier()
modeldc.fit(X_train,Y_train)
modeldc.score(X_train,Y_train)
modeldc.score(X_test,Y_test)
Y_pred=modeldc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
classification_report(Y_pred,Y_test)
confusion_matrix(Y_pred,Y_test)
modeldc.get_params()
from sklearn.model_selection import GridSearchCV
param_grid = {'criterion' :['gini', 'entropy'],'max_depth': [4,6,8,12]}
search = GridSearchCV(modeldc, param_grid, cv=5)
search.fit(X_train,Y_train)
search.best_estimator_.get_params()
search.score(X_test,Y_test)
