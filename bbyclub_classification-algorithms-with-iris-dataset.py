# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings 

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#2.1. Data Import

data = pd.read_csv('../input/Iris.csv')
data.head()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder() #Label Encoder sınıfından bir nesne tanımlıyoruz





#Visualization

X=data.iloc[:,1:3] #2d data

y=data.iloc[:,-1:]

y.iloc[:,0]=le.fit_transform(y.iloc[:,0]) # Now y is dataframe and we have to convert it into numpy array

y=y.values.astype(int) #convert y into numpy array

y_new = y.reshape((150,)) #reshape array from (150,1) to (150,) because we use y as color spectral

print(type(y_new))

print((y_new.shape))



x_min, x_max= X.iloc[:,0].min() - .5, X.iloc[:,0].max() + .5

y_min, y_max= X.iloc[:,1].min() - .5, X.iloc[:,1].max() + .5



plt.figure(2, figsize=(8,6))

plt.clf()





#plot the training points

plt.scatter(X.iloc[:,0], X.iloc[:,1], c =y_new, cmap=plt.cm.Set1, edgecolor="k")

plt.xlabel("Sepal length")

plt.ylabel("Sepal width")



plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xticks(())

plt.yticks(())



fig= plt.figure(1, figsize=(8,6))

ax= Axes3D(fig, elev=-150, azim=110)

ax.scatter(data.iloc[:,1],data.iloc[:,2],data.iloc[:,3], c=y_new, cmap=plt.cm.Set1, edgecolor="k", s=40)

ax.set_title("IRIS Verisi")

ax.set_xlabel("birinci ozellik")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("İkinci Ozellik")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("Ucuncu Ozellik")

ax.w_zaxis.set_ticklabels([])



plt.show()
x = data.iloc[:,1:5].values #independent variables

y = data.iloc[:,5:].values #dependent variable

print(y)
#Test and Training Split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
#Data scaling

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.transform(x_test)
# Classification algorithms start from here

# 1. Logistic Regression



from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(X_train,y_train) #train



y_pred = logr.predict(X_test) #prediction

print(y_pred)

print(y_test)
#Confusion matrix and Accuracy ratio

cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy of Logistic Regression algo: ",logr.score(X_test,y_test))
# 2. KNN



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

knn.fit(X_train,y_train)



y_pred = knn.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print(cm)

print("Accuracy of KNN algo: ",knn.score(X_test,y_test))
# 3. SVC (SVM classifier)

from sklearn.svm import SVC

svc = SVC(kernel='rbf') #rfb of poly

svc.fit(X_train,y_train)



y_pred = svc.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print('SVC')

print(cm)

print("Accuracy of SVM algo: ",svc.score(X_test,y_test))
# 4. NAive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)



y_pred = gnb.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print('GNB')

print(cm)

print("Accuracy of Naive Bayes algo: ",gnb.score(X_test,y_test))


# 5. Decision tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy')



dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print('DTC')

print(cm)

print("Accuracy of Decision Tree algo: ",dtc.score(X_test,y_test))
# 6. Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')

rfc.fit(X_train,y_train)



y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print('RFC')

print(cm)

print("Accuracy of Random Forest algo: ",rfc.score(X_test,y_test))
# 7. ROC , TPR, FPR değerleri 



y_proba = rfc.predict_proba(X_test)

print(y_test)

print(y_proba[:,0])



from sklearn import metrics

fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='Iris-virginica')

print(fpr)

print(tpr)