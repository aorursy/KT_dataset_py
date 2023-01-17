# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.info()
corr_mat=df.corr()
corr_mat['quality'].sort_values(ascending =True)
corr_mat
plt.figure(figsize=(20,5))
sns.heatmap(df.corr(),annot=True)
df.drop(['residual sugar','free sulfur dioxide','pH'],axis=1,inplace=True)
df['quality'].unique()
bins=(2,6.5,8)
group_names=['bad','good']
df['quality']=pd.cut(df['quality'],bins=bins,labels=group_names)
df.head()
from sklearn.preprocessing import LabelEncoder
label_quality=LabelEncoder()
df['quality']=label_quality.fit_transform(df['quality'])
df.head()
X=df.drop('quality',axis=1)
y=df['quality']
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=5,test_size=0.25,random_state=42)
for train_index,test_index in split.split(X,y):
    X_train,X_test=X.loc[train_index],X.loc[test_index]
    y_train,y_test=y.loc[train_index],y.loc[test_index]
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=42,n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,pred_rfc))
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(random_state=42)
sgd.fit(X_train,y_train)
pred_sgd=sgd.predict(X_test)
print(classification_report(y_test,pred_sgd))
from sklearn.svm import SVC
svc=SVC(random_state=42)
svc.fit(X_train,y_train)
pred_svc=svc.predict(X_test)
print(classification_report(y_test,pred_svc))
from sklearn.model_selection import  GridSearchCV
params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "penalty" : ["l2", "l1", "none"],
    "alpha": [10 ** x for x in range(-6, 1)]
}
grid_sgd = GridSearchCV(estimator=sgd, param_grid=params, scoring='f1', cv=10)
grid_sgd.fit(X_train, y_train)
grid_sgd.best_params_
sgd2 = SGDClassifier(alpha = 1, loss =  'modified_huber', penalty='none')
sgd2.fit(X_train, y_train)
pred_sgd2 = sgd2.predict(X_test)
print(classification_report(y_test, pred_sgd2))
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(estimator=svc, param_grid=param, scoring='f1', cv=10)
grid_svc.fit(X_train,y_train)
grid_svc.best_params_
svc2 = SVC(C= 1.4, gamma= 0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
rfc2 = RandomForestClassifier(criterion= 'gini',
 max_depth= 8,
 max_features='log2',
 n_estimators= 200)
rfc2.fit(X_train, y_train)
pred_rfc2 = rfc2.predict(X_test)
print(classification_report(y_test, pred_rfc2))
from sklearn.model_selection import cross_val_score
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()
svc_eval = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
svc_eval.mean()
sgd_eval = cross_val_score(estimator = sgd, X = X_train, y = y_train, cv = 10)
sgd_eval.mean()
