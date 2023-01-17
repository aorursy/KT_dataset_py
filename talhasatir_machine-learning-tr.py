# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data =pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.tail()
data.diagnosis.value_counts()
data.diagnosis =[1 if each =='B'else 0 for each in data.diagnosis] #y kolonu yapacagım kolonu int degerlere cevirdim
data.tail()
data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
x,y=data.loc[:,data.columns !='diagnosis'],data.loc[:,data.columns =='diagnosis'] 
data.columns
x
from sklearn.model_selection import train_test_split #verimi egitim test diye böldüm

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=42)
#Logistic Regressor

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

y_head=lr.predict(x_test)

lr.score(x_test,y_test) #gerçek degerler ve tahminler arası dogruluk oranı
#KNN

from sklearn.neighbors import KNeighborsClassifier

knn =KNeighborsClassifier(n_neighbors =10)

knn.fit(x_train,y_train)

knn.score(x_test,y_test)
score_list=[]

for i in range(1,20):

    knn2=KNeighborsClassifier(n_neighbors =i)

    knn2.fit(x_train,y_train)

    ss=knn2.score(x_test,y_test)

    score_list.append(ss)



plt.figure(figsize=[10,6])

plt.plot(range(1,20),score_list)

plt.xlabel('score_value')

plt.ylabel('score output')

plt.show()

    
#SVM

from sklearn.svm import SVC

svm =SVC(random_state=12)

svm.fit(x_train,y_train)

y_head=svm.predict(x_test)

svm.score(x_test,y_test)
#Naive_Bayes

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

y_head=nb.predict(x_test)

nb.score(x_test,y_test)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_head =dt.predict(x_test)

dt.score(x_test,y_test)
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf =RandomForestClassifier(n_estimators=100,random_state=12)

rf.fit(x_train,y_train)

y_head=rf.predict(x_test)

rf.score(x_test,y_test)

score_list2 =[]

for i in range (2,250):

    rf2 =RandomForestClassifier(n_estimators=i,random_state=12)

    rf2.fit(x_train,y_train)

    ll=rf2.score(x_test,y_test)

    score_list2.append(ll)



plt.figure(figsize=[13,7])

plt.plot(range(2,250),score_list2)

plt.xlabel('score_value')

plt.ylabel('output_score')

plt.show()
#Random Forest_Confusion Matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_head)

cm
import seaborn as sns

f,ax =plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot =True,linewidths=0.5,linecolor='red',fmt='.0f',ax=ax)

plt.xlabel('y_pred')

plt.ylabel('y_true')

plt.show()