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
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df
df.isnull().sum()
df.corr()['Outcome']
y=df['Outcome']
x=df.drop(['Outcome'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_s=knn.predict(x_test)

    scores=accuracy_score(y_test,pred_s)

    list_1.append(scores)

    
import matplotlib.pyplot as plt

plt.plot(range(1,11),list_1)

plt.xlabel('k values')

plt.ylabel('accuracy scores')

plt.show()
#from figure k=9 gives the best accuracy score
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=1000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier()

grid_param = {

    'n_estimators': [100,150,200,250,300],

    'criterion': ['gini', 'entropy'],

    'bootstrap': [True, False]

}
clf=GridSearchCV(rfc,grid_param,cv=5,verbose=0)
clf.fit(x_train,y_train)
print(clf.best_params_)
rfc=RandomForestClassifier(bootstrap=True,criterion='entropy',n_estimators=250)
rfc.fit(x_train,y_train)

pred_3=rfc.predict(x_test)
score_3=accuracy_score(y_test,pred_3)
score_3
from sklearn.svm import SVC

svm=SVC()

params= {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['rbf']}  

clf_2=GridSearchCV(svm,params,cv=5,verbose=0)
clf_2.fit(x_train,y_train)
print(clf_2.best_params_)
svm=SVC(C=1,gamma=0.0001,kernel='rbf')

svm.fit(x_train,y_train)

pred_4=svm.predict(x_test)
score_4=accuracy_score(y_test,pred_4)
score_4
#from all classifiers logistic regression gives the best accuracy score