# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dftrain=pd.read_csv('/kaggle/input/titanic/train.csv')

dftest=pd.read_csv('/kaggle/input/titanic/test.csv')
dftrain['Age']=dftrain['Age'].fillna(22)

dftest['Age']=dftest['Age'].fillna(22)

dftrain['Embarked']=dftrain['Embarked'].fillna('S')

dftest['Fare']=dftest['Fare'].fillna(0)

dftrain.head()
dftest.head()
X_train=dftrain.values

y_train=X_train[:,1]

'''X_train=np.delete(X_train, 0, 1)#PassengerId

X_train=np.delete(X_train, 1, 1)#Survied (Label)

X_train=np.delete(X_train, 2, 1)#Name

X_train=np.delete(X_train, 6, 1)#Ticket

X_train=np.delete(X_train, 7, 1)#Cabin

#X_test=np.delete(X_test, 7, 1)#Embarked

X_train'''

X_train=X_train[:,[2,4,5,6,7,9]]

X_test=dftest.values



X_test=X_test[:,[1,3,4,5,6,8]]

X_test
X_train[:, 1] =np.where(X_train[:, 1] =='male', 0,1)

X_test[:, 1] =np.where(X_test[:, 1] =='male', 0,1)
#from sklearn.neighbors import KNeighborsClassifier

#knnclf = KNeighborsClassifier(n_neighbors=3)

#neigh.fit(X_train, y_train.astype('int')) 

#result=neigh.predict(X_test)



#from sklearn.linear_model import LogisticRegression

#clflr = LogisticRegression(solver='liblinear')



#from sklearn.svm import SVC

#svcclf = SVC(gamma='auto')



#from sklearn.tree import DecisionTreeClassifier

#dtclf = DecisionTreeClassifier()

#clf = clf.fit(X_train,y_train.astype('int'))'''







from sklearn.ensemble import RandomForestClassifier, VotingClassifier

rfclf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)

rfclf = rfclf.fit(X_train,y_train.astype('int'))



#eclf1 = VotingClassifier(estimators=[('knn', knnclf), ('dt', dtclf), ('rf', rfclf),('lr',clflr),('svc',svcclf)], voting='hard')

#eclf1 = eclf1.fit(X_train,y_train.astype('int'))

y_pred=rfclf.predict(X_test)
ids = dftest["PassengerId"].to_list()

file = open("answer.csv", "w")

file.write("PassengerId,Survived\n")

for id_, pred in zip(ids, y_pred):

    file.write("{},{}\n".format(id_, pred))

file.close()