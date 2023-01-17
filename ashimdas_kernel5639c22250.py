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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import itertools



diab=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

diab.head(5)
diab.isnull().sum()
sns.countplot(x='Outcome',data=diab)

plt.show()
columns=diab.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    diab[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
diab1=diab[diab['Outcome']==1]

columns=diab.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    diab1[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
sns.pairplot(data=diab,hue='Outcome',diag_kind='kde')

plt.show()
sns.heatmap(diab[diab.columns[:8]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegression
outcome=diab['Outcome']

data=diab[diab.columns[:8]]

train,test=train_test_split(diab,test_size=0.25,random_state=0,stratify=diab['Outcome'])# stratify the outcome

train_X=train[train.columns[:8]]

test_X=test[test.columns[:8]]

train_Y=train['Outcome']

test_Y=test['Outcome']

train_X.head(10)
from sklearn import metrics

types=['rbf','linear']

for i in types:

    model=svm.SVC(kernel=i)

    model.fit(train_X,train_Y)

    prediction=model.predict(test_X)

    print('Accuracy for SVM kernel=',i,'is',metrics.accuracy_score(prediction,test_Y))
model = LogisticRegression()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))
model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_Y))
a_index=list(range(1,11))

a=pd.Series()

x=[0,1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_Y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))

plt.plot(a_index, a)

plt.xticks(x)

plt.show()

print('Accuracies for different values of n are:',a.values)


abc=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=3),DecisionTreeClassifier()]

for i in models:

    model = i

    model.fit(train_X,train_Y)

    prediction=model.predict(test_X)

    abc.append(metrics.accuracy_score(prediction,test_Y))

models_dataframe=pd.DataFrame(abc,index=classifiers)   

models_dataframe.columns=['Accuracy']

models_dataframe