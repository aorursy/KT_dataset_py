# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
dataset = pd.read_csv("../input/train.csv")
test2= pd.read_csv("../input/test.csv")
test2.info()
dataset.head()
dataset.isna()
dataset.head(7)
dataset.head()
sex = pd.get_dummies(dataset['Sex'],drop_first= True)
                     
embarked = pd.get_dummies(dataset['Embarked'],drop_first= True)
sex.head()
embarked.head()
dataset.head()
pclass = pd.get_dummies(dataset['Pclass'],drop_first= True)
dataset.head()

pclass.head()
dataset.drop(['PassengerId','Name','Embarked','Sex','Ticket','Cabin','Pclass'],axis=1,inplace=True)

dataset.head()
dataset = pd.concat([dataset,embarked,pclass,sex],axis =1 )
dataset.head()
dataset.dropna(inplace=True)
dataset.isnull().sum()
X = dataset.iloc[:,1:10].values
Y = dataset.iloc[:,0].values
X
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .3)
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression 

clf = LogisticRegression()


clf.fit(X_train,Y_train)



predict = clf.predict(X_test)

from sklearn.metrics import confusion_matrix as cm
cm(Y_test,predict)
from sklearn.metrics import accuracy_score as ac
ac(Y_test,predict)

from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(X_train, Y_train) 
y_predsvm = clf.predict(X_test)

from sklearn.metrics import confusion_matrix as cm
cm(Y_test,y_predsvm)
from sklearn.metrics import accuracy_score as ac
ac(Y_test,y_predsvm)
