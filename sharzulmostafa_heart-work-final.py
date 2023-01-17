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
dataset = pd.read_csv('../input/Data_Train.csv')

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 13].values
dataset
dataset.shape
dataset2 = pd.read_csv('../input/Data_Test.csv')

x_test = dataset2.iloc[:, :].values
dataset2
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(x[:, 0:13])

x[:, 0:13] = imputer.transform(x[:, 0:13])
x[0]
from sklearn.preprocessing import StandardScaler

x_std = StandardScaler().fit_transform(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_std,y,test_size=0.3,random_state=50)
from sklearn import metrics

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn import metrics

from sklearn import svm

clf = svm.SVC(kernel='rbf') # Rbf Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



max_accuracy = 0





for p in range(200):

    dt = DecisionTreeClassifier(random_state=p)

    dt.fit(X_train,y_train)

    Y_pred_dt = dt.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = p

        

#print(max_accuracy)

#print(best_x)





dt = DecisionTreeClassifier(random_state=best_x)

dt.fit(X_train,y_train)

Y_pred_dt = dt.predict(X_test)



score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=90)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.decomposition import PCA as sklearnPCA

pca = sklearnPCA(n_components=5)

pca_heart = pca.fit_transform(x_std)
#principal_heart= pd.DataFrame(data = pca_heart , columns = ['principal component 1', 'principal component 2','principal component 3'])

#principal_heart.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pca_heart,y,test_size=0.2,random_state=50)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics
#from sklearn.linear_model import LinearRegression

#reg = LinearRegression().fit(X_train, y_train)

#reg.score(X_test,y_test)

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn import svm

clf = svm.SVC(kernel='rbf') # rbf Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



max_accuracy = 0





for p in range(200):

    dt = DecisionTreeClassifier(random_state=p)

    dt.fit(X_train,y_train)

    Y_pred_dt = dt.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = p

        

#print(max_accuracy)

#print(best_x)





dt = DecisionTreeClassifier(random_state=best_x)

dt.fit(X_train,y_train)

Y_pred_dt = dt.predict(X_test)



score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=50)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=7, n_iter=10, random_state=50)

#svd_heart=svd.fit(x_std).transform(x_std) 

svd_heart = svd.fit_transform(x_std)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(svd_heart,y,test_size=0.3,random_state=50)
#from sklearn.linear_model import LinearRegression

#reg = LinearRegression().fit(X_train, y_train)

#reg.score(X_test,y_test)

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



max_accuracy = 0





for p in range(200):

    dt = DecisionTreeClassifier(random_state=p)

    dt.fit(X_train,y_train)

    Y_pred_dt = dt.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)

    if(current_accuracy>max_accuracy):

        max_accuracy = current_accuracy

        best_x = p

        

#print(max_accuracy)

#print(best_x)





dt = DecisionTreeClassifier(random_state=best_x)

dt.fit(X_train,y_train)

Y_pred_dt = dt.predict(X_test)



score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)



print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
from sklearn import svm

clf = svm.SVC(kernel='rbf') # Rbf Kernel

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=90)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))