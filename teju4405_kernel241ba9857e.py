# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
placement=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
placement.fillna(0,inplace=True)
placement.head()
scatter_matrix(placement,figsize=(10,10))
placement['workex']=pd.get_dummies(placement['workex'])
placement['degree_t']=pd.get_dummies(placement['degree_t'])
placement['gender']=pd.get_dummies(placement['gender'])
placement['specialisation']=pd.get_dummies(placement['specialisation'])
placement.head()
placement=placement.drop(['ssc_b','hsc_b'],axis=1)
placement.head()
placement['ssc_p']=(placement['ssc_p']-placement['ssc_p'].mean())/placement['ssc_p'].std()
placement['hsc_p']=(placement['hsc_p']-placement['hsc_p'].mean())/placement['hsc_p'].std()
placement['degree_p']=(placement['degree_p']-placement['degree_p'].mean())/placement['degree_p'].std()
placement['etest_p']=(placement['etest_p']-placement['etest_p'].mean())/placement['etest_p'].std()
placement['mba_p']=(placement['mba_p']-placement['mba_p'].mean())/placement['mba_p'].std()
placement.head()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
neigh=KNeighborsClassifier(n_neighbors=4)
lr=LogisticRegression(C=0.01,solver='sag')
sv=svm.SVC(kernel='rbf')
x=placement[['gender','ssc_p','hsc_p','degree_p','degree_t','workex','etest_p','specialisation','mba_p']]
y=placement['status']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
print(x.shape,y.shape)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
neigh.fit(x_train,y_train)
lr.fit(x_train,y_train)
sv.fit(x_train,y_train)
neigh.predict(x_test)
print(metrics.accuracy_score(y_test,neigh.predict(x_test)))
ks=15
mean=np.zeros((ks-1))
std=np.zeros((ks-1))
Confusion=[];
for n in range(1,ks):
    neigh1=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    yht=neigh1.predict(x_test)
    mean[n-1]=metrics.accuracy_score(y_test,yht)
    std[n-1]=np.std(yht==y_test)/np.sqrt(yht.shape[0])
mean
plt.plot(range(1,ks),mean,'g')
plt.fill_between(range(1,ks),mean-1*std,mean+1*std,alpha=1.2)
print('Max Accuracy is ',mean.max(),'with k =',mean.argmax()+1)
from sklearn.metrics import classification_report
print (classification_report(y_test, yht))

lr.predict(x_test)
print(metrics.accuracy_score(y_test,lr.predict(x_test)))
yhat=lr.predict(x_test)
at=lr.predict_proba(x_test)
from sklearn.metrics import log_loss
log_loss(y_test,at)
print (classification_report(y_test, yhat))

sv.predict(x_test)
print(metrics.accuracy_score(y_test,sv.predict(x_test)))
hat=sv.predict(x_test)
print(classification_report(y_test,hat))

