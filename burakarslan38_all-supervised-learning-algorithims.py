# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split







from sklearn.metrics import confusion_matrix

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/hepatitis.data.txt")

#df.info()

df.replace("?",-99999,inplace=True)

df = df.convert_objects(convert_numeric=True)
x = df.drop(["class"],axis=1)

y = df["class"].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_test,y_test)

accuracy_lor=lr.score(x_train,y_train)

print("linear regression accuracy:%{}".format(accuracy_lor*100))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()

dtr.fit(x_test,y_test)

accuracy_dtr = dtr.score(x_train,y_train)

print("decision tree regression accuracy:%{}".format(accuracy_dtr*100))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(x_test,y_test)

accuracy_rfr = rfr.score(x_train,y_train)

print("random forest regression accuracy:%{}".format(accuracy_rfr*100))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

accuracy_lr=lr.score(x_test,y_test)

print("logistic regression accuracy:%{}".format(accuracy_lr*100))


########################################

#Choosing best k value

from  sklearn.neighbors import KNeighborsClassifier

from collections import Counter

scores2={}

scores=[]

index=[]

def sort(scores):        

    votes = [i for i in sorted(scores)]

    print(scores)

    vote_result = max(scores)

    return vote_result



for each in range(1,18):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score=(knn2.score(x_test,y_test))

    scores2.update({each:score})

    scores.append(knn2.score(x_test,y_test))

    #index.append(each)

vote_result=sort(scores2)  

print(vote_result)

########################################

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11,n_jobs=-1)#k default 3, n_jobs -1 for max performance

knn.fit(x_train,y_train)

accuracy_knn = knn.score(x_test,y_test)

print("knn accuracy:%{}".format(accuracy_knn*100))

########################################

plt.plot(range(1,18),scores)

plt.xlabel("k values")

plt.ylabel("scores")

plt.show()
from sklearn.svm import SVC

svm = SVC(random_state=1)

svm.fit(x_train,y_train)

accuracy_svm = svm.score(x_test,y_test)

print("svm accuracy:%{}".format(accuracy_svm*100))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

accuracy_nb = nb.score(x_test,y_test)

print("naive bayes accuracy:%{}".format(accuracy_nb*100))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

accuracy_dt = dt.score(x_test,y_test)

print("decision tree accuracy:%{}".format(accuracy_dt*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000)#n_estimators --> tree sayısı

rf.fit(x_train,y_train)

accuracy_rf = rf.score(x_test,y_test)

print("random forest accuracy:%{}".format(accuracy_rf*100))
print("linear regression accuracy:%{}".format(accuracy_lor*100))

print("decision tree regression accuracy:%{}".format(accuracy_dtr*100))

print("random forest regression accuracy:%{}".format(accuracy_rfr*100))

print("logistic regression accuracy:%{}".format(accuracy_lr*100))

print("knn accuracy:%{}".format(accuracy_knn*100))

print("svm accuracy:%{}".format(accuracy_svm*100))

print("naive bayes accuracy:%{}".format(accuracy_nb*100))

print("decision tree accuracy:%{}".format(accuracy_dt*100))

print("random forest accuracy:%{}".format(accuracy_rf*100))
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

models=[]

models.append(("LR",LogisticRegression()))

models.append(("NB",GaussianNB()))

models.append(("KNN",KNeighborsClassifier(n_neighbors=5)))

models.append(("DT",DecisionTreeClassifier()))

models.append(("SVM",SVC()))

for name, model in models:

    

    clf=model



    clf.fit(x_train, y_train)



    y_pred=clf.predict(x_test)

    print(10*"=","{} için Sonuçlar".format(name).upper(),10*"=")

    print("Başarı oranı:{:0.2f}".format(accuracy_score(y_test, y_pred)))

    print("Karışıklık Matrisi:\n{}".format(confusion_matrix(y_test, y_pred)))

    print("Sınıflandırma Raporu:\n{}".format(classification_report(y_test,y_pred)))

    print(30*"=")