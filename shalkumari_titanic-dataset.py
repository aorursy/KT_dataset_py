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
train = pd.read_csv("../input/train.csv")
test =pd.read_csv("../input/test.csv")
train.head()
print(train.info())
print(test.info())
print(train.describe())
train['Age'].fillna(train['Age'].median() , inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace= True)
test['Age'].fillna(test['Age'].median() , inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace= True)
print(train.info())
print(test.info())
train['FamilySize']=train['SibSp']+train['Parch']
test['FamilySize']=test['SibSp']+test['Parch']
drop=['PassengerId' ,'Ticket' ,'Cabin' ,'Parch', 'SibSp', 'Name','Survived']
X=train.copy()
X.drop(drop, axis=1, inplace=True)
drop1=['PassengerId' ,'Ticket' ,'Cabin' ,'Parch', 'SibSp', 'Name']
X_test=test.copy()
X_test.drop(drop1, axis=1, inplace=True)
Y=train['Survived']
from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()
X['Sex']=labelencoder.fit_transform(X['Sex'])
X_test['Sex']=labelencoder.fit_transform(X_test['Sex'])
X['Embarked']=labelencoder.fit_transform(X['Embarked'])
X_test['Embarked']=labelencoder.fit_transform(X_test['Embarked'])
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier=classifier.fit(X,Y)
Y_pred=classifier.predict(X_test)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 25,5
import matplotlib.pyplot as plt
K=pd.Series(range(1,101))
plt.scatter(K,Y[0:100], color='red')
plt.scatter(K,classifier.predict(X)[0:100], color='blue')
Y_pred1=classifier.predict(X)
e=0
for i in range(0,891):
    if (Y[i]-Y_pred1[i]!=0):
        e=+1
accuracy=((891-e)/891)*100 
print('accuracy= ',accuracy,'%')
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn=knn.fit(X,Y)
import matplotlib.pyplot as plt
K=pd.Series(range(1,101))
plt.scatter(K,Y[0:100], color='red')
plt.scatter(K,knn.predict(X)[0:100], color='blue')
Y_pred1=knn.predict(X)
e=0
for i in range(0,891):
    if (Y[i]-Y_pred1[i]!=0):
        e=+1
accuracy=((891-e)/891)*100 
print('accuracy= ',accuracy,'%')
prediction= pd.DataFrame(Y_pred,columns=['Survived'])
prediction= concat([test['PassengerID'],prediction], axis=1)
