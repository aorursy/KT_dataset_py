
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")
data2=pd.read_csv("../input/test.csv")
data.head()
data.describe()
#data['Embarked']=data['Embarked'].fillna('S')
data.fillna(data.mean(),inplace=True)
data["Sex"][data["Sex"] == "male"] = 0
data["Sex"][data["Sex"]=='female'] = 1
data2.fillna(data.mean(),inplace=True)
data2["Sex"][data2["Sex"] == "male"] = 0
data2["Sex"][data2["Sex"]=='female'] = 1
#data["Embarked"][data["Embarked"] =='S'] = 0
#data["Embarked"][data["Embarked"]=='C'] = 1
#data["Embarked"][data["Embarked"]=='Q'] = 2
x=data[['Age','Sex','Pclass']]
y=data[['Survived']]
x_test=data2[['Age','Sex','Pclass']]
#from sklearn.cross_validation import train_test_split
#from sklearn import metrics
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
#from sklearn import svm
#x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2)
#k_scores=[]
#k_range=range(1,31)
#for i in k_range:
 #  knn=KNeighborsClassifier(n_neighbors=i)
  # knn.fit(x_train,y_train)
  # y_pred=knn.predict(x_test)
  # k_scores.append(metrics.accuracy_score(y_test,y_pred,normalize=True))
#k_range=range(1,31)
#for i in k_range:
 #  knn=KNeighborsClassifier(n_neighbors=i)
  # knn.fit(x_train,y_train)
  # y_pred=knn.predict(x_test)
  # k_scores.append(metrics.accuracy_score(y_test,y_pred,normalize=True))
#k_mean=0;
#for i in k_scores:
 #   k_mean=k_mean+i
#k_mean=k_mean/30
my_tree=tree.DecisionTreeClassifier()
my_tree=my_tree.fit(x,y)
my_pred=my_tree.predict(x_test)
print(my_pred)
#tree_score=metrics.accuracy_score(y_test,my_pred,normalize=True)

#svmc=svm.SVC()
#svmc.fit(x_train,y_train)
#pred=svmc.predict(x_test)
#svm_score=metrics.accuracy_score(y_test,pred,normalize=True)
#knn2=KNeighborsClassifier(n_neighbors=6)
#knn2.fit(x_train,y_train)
#y_pred2=knn.predict(x_test)
#knn_score=metrics.accuracy_score(y_test,y_pred,normalize=True)

#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.plot(k_range,k_scores)
#plt.xlabel("K in KNN")
#plt.ylabel("score")



PassengerId =np.array(data2["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_pred,PassengerId,columns = ["Survived"])
my_solution.index.name='PassengerId'
print(my_solution)
my_solution.to_csv('submission.csv', index="PassengerId")
