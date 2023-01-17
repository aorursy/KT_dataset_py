import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv('../input/breastcancer.csv')
data.head()
data.tail()
data.columns
data.isnull().sum()
data.shape
data.describe()
data.groupby(['diagnosis']).size()
sns.countplot(x='diagnosis',data=data)
sns.pairplot(data=data,hue='diagnosis',vars=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean'])
sns.scatterplot(data=data,x='area_mean',y='smoothness_mean',hue='diagnosis')
plt.figure(figsize=(20,10))

sns.heatmap(data.corr(),annot=True,cmap='Blues')

plt.show()
X=data.drop(['diagnosis','id'],axis=1)

y=data['diagnosis']
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report



lr=LogisticRegression(solver='lbfgs')

lr.fit(X_train,y_train)

y_predl=lr.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy of training set',accuracy_score(y_train,lr.predict(X_train)))

print('Accuracy of testing set ',accuracy_score(y_test,y_predl))

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_predd=clf.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy of training set',accuracy_score(y_train,clf.predict(X_train)))

print('Accuracy of testing set ',accuracy_score(y_test,y_predd))

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(X_train,y_train)

y_predr=rf.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy of training set',accuracy_score(y_train,rf.predict(X_train)))

print('Accuracy of testing set ',accuracy_score(y_test,y_predr))

from sklearn.svm import SVC

sv=SVC()

sv.fit(X_train,y_train)

y_preds=sv.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy of training set',accuracy_score(y_train,sv.predict(X_train)))

print('Accuracy of testing set ',accuracy_score(y_test,y_preds))

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

knn.fit(X_train,y_train)

y_predk=knn.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy of training set',accuracy_score(y_train,knn.predict(X_train)))

print('Accuracy of testing set ',accuracy_score(y_test,y_predk))
from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,y_preds)

cm
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_preds))
min_train=X_train.min()

min_train
new_train=(X_train-min_train).max()

X_train_scaled=(X_train-min_train)/new_train

X_train_scaled
sns.scatterplot(x=X_train['area_mean'],y=X_train['smoothness_mean'],hue=y_train)
sns.scatterplot(x=X_train_scaled['area_mean'],y=X_train_scaled['smoothness_mean'],hue=y_train)
min_test=X_test.min()

new_test=(X_test-min_test).max()

X_test_scaled=(X_test-min_test)/new_test

X_test_scaled
sv=SVC()

sv.fit(X_train_scaled,y_train)

y_preds=sv.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

print('Accuracy of training set',accuracy_score(y_train,sv.predict(X_train_scaled)))

print('Accuracy of testing set ',accuracy_score(y_test,y_preds))
cm=confusion_matrix(y_test,y_preds)

sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_preds))
param_grid={'C':[0.1,1,10,50,100],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_
grid_pred=grid.predict(X_test_scaled)
cm=confusion_matrix(y_test,grid_pred)

cm

sns.heatmap(cm,annot=True)
print(classification_report(y_test,grid_pred))