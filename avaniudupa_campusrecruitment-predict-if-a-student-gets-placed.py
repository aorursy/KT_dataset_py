# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv",index_col='sl_no')
data.head()
x=data.iloc[:,:12].values

y=data.iloc[:,-2].values
data.isnull().sum()
data['ssc_b'].unique()
data['hsc_b'].unique()
data['hsc_s'].unique()
data['degree_t'].unique()
data['specialisation'].unique()
from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder()

x=encoder.fit_transform(x)
from sklearn.preprocessing import LabelEncoder

enc=LabelEncoder()

y=enc.fit_transform(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1)



rf.fit(x_train,y_train)

accuracy=rf.score(x_test,y_test)



print("Random Forest accuracy is :{}".format(accuracy))
from sklearn.naive_bayes import MultinomialNB

# Instantiate the classifier

mnb = MultinomialNB()



# Train classifier

mnb.fit( x_train,y_train)

accuracy=mnb.score(x_test,y_test)

print("Naive bayes accuracy is :{}".format(accuracy))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=29)            #n_neighbors optimal value should be suqare root of n

knn.fit(x_train,y_train)

y_pred_knn=knn.predict(x_test)



#finding accuracy and confusion matrix

accuracy=knn.score(x_test,y_test)

print("KNN accuracy is :{}".format(accuracy))
from sklearn import svm    			

C = 0.6  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)

#svc = svm.LinearSVC(C=C).fit(X, y)

rbf_svc = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(x_train, y_train)

# SVC with polynomial (degree 3) kernel

poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(x_train, y_train)

accuracy1=svc.score(x_test,y_test)

print("SVM accuracy is :{}".format(accuracy1))

accuracy2=rbf_svc.score(x_test,y_test)

print("SVM rbf accuracy is :{}".format(accuracy2))

accuracy3=poly_svc.score(x_test,y_test)

print("SVM poly accuracy is :{}".format(accuracy3))
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth = 6,random_state = 99, max_features = None, min_samples_leaf = 5)

dtree.fit(x_train,y_train)

accuracy=dtree.score(x_test,y_test)

print("Decision tree accuracy is :{}".format(accuracy))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

accuracy=lr.score(x_test,y_test)

print("Decision tree accuracy is :{}".format(accuracy))

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(x,y)
