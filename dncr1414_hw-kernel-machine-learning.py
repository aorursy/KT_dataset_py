# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/column_2C_weka.csv")
#ön analiz - preliminary analysis

data.head()
#ön analiz - preliminary analysis

data.info()
#ön analiz - preliminary analysis

data.describe()
# string to int..

data.iloc[:,-1]=[1 if each == "Abnormal" else 0 for each in data.iloc[:,-1]]
y=data.iloc[:,-1].values.reshape(-1,1)

x_data=data.drop(["class"],axis=1)
#Normalizition

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#Train Tesp Split

from sklearn.model_selection import train_test_split



x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=29)



#Logistic Regression

from sklearn.linear_model import LogisticRegression



lr=LogisticRegression()

lr.fit(x_train,y_train)



print ("test accuracy {}".format(lr.score(x_test,y_test)))
#KNN k=5

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

print ("test accurary {}".format(knn.score(x_test,y_test)))
#KNN k=17

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=17)

knn.fit(x_train,y_train)

print ("test accurary {}".format(knn.score(x_test,y_test)))
#Support Vector Machine

from sklearn.svm import SVC



svm=SVC(random_state=29)

svm.fit(x_train,y_train)



print ("Test Accurary {}".format(svm.score(x_test,y_test)))
#Bayes

from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()

nb.fit(x_train,y_train)



print("Test Accurary {}".format(nb.score(x_test,y_test)))
#Decision Tree 



from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()



dt.fit(x_train,y_train)



print("Test Accurary {}".format(dt.score(x_test,y_test)))
#Random Forest 

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=50,random_state=29)



rfc.fit(x_train,y_train)



print("Test Accurary {}".format(rfc.score(x_test,y_test)))