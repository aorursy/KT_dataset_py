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
df=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.dtypes
#df['specialisation'].value_counts()
df.drop('salary',axis=1,inplace=True)
df['gender'].replace({'M':0,'F':1},inplace=True)
df['hsc_b'].replace({"Others":0,'Central':1},inplace=True)
df['hsc_s'].replace({'Commerce':0,'Science':1,'Arts':2},inplace=True)
df['degree_t'].replace({'Comm&Mgmt':0,'Sci&Tech':2,'Others':1},inplace=True)
df['workex'].replace({"No":0,'Yes':1},inplace=True)
df['specialisation'].replace({'Mkt&Fin':1,'Mkt&HR':0},inplace=True)
df.drop('sl_no',axis=1,inplace=True)
df['ssc_b'].replace({"Others":0,'Central':1},inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
X_train,X_test,y_train,y_test=train_test_split(df.drop('status',axis=1),df['status'],random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
forest=RandomForestClassifier(n_estimators=1000,random_state=0,max_depth=5)
forest.fit(X_train,y_train)
y_pred=forest.predict_proba(X_test)
y_pred=(y_pred[:,1]>0.75).astype(int)  # if probaility of getting placed is more than 0.75 then 1 otherwise 0
#y_test=y_test.replace({'Not Placed':0,'Placed':1})
print(accuracy_score(y_test,y_pred)) # gets you upto 0.77 accuracy score
print(pd.Series(y_pred).value_counts())
print(y_test.value_counts())
from sklearn.model_selection import GridSearchCV
clf=RandomForestClassifier()
grid=GridSearchCV(clf,{'max_depth':[2,3,4,5,6,7,8,9,10],'n_estimators':[10,50,100,125,150]},scoring='accuracy')
grid.fit(X_train,y_train)
print(grid.best_params_)
final_clf=grid.best_estimator_
y_pred_train=final_clf.predict(X_train)
print(accuracy_score(y_train,y_pred))
y_pred_test=final_clf.predict(X_test)
accuracy_score(y_test,y_pred_test)
svc=SVC()
grid1=GridSearchCV(svc,{'C':[0.01,0.1,1,10,100],'gamma':[0.01,0.1,1,10,100],'kernel':['linear','rbf']},scoring='accuracy')
grid1.fit(X_train,y_train)
print(grid1.best_params_)
final_svc=grid1.best_estimator_

print(accuracy_score(y_test,final_svc.predict(X_test)))#test accuracy
print(accuracy_score(y_train,final_svc.predict(X_train)))#train accuracy
#This model is clearly overfitting

svc=SVC(C=0.1,kernel='linear')
svc.fit(X_train,y_train)
print(accuracy_score(y_train,svc.predict(X_train)))#train accuracy
print(accuracy_score(y_test,svc.predict(X_test)))#test accuracy
precision_score(pd.Series(y_test).replace({'Not Placed':0,'Placed':1}),pd.Series(svc.predict(X_test)).replace({'Not Placed':0,'Placed':1}))
