# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#For importing all the libraries required.



import pandas as pd

import numpy as np



import seaborn as sns

import random as rnd

import missingno as ms

import matplotlib.pyplot as plt

%matplotlib inline



sns.set()



#Gathering the data and storing them using pandas.

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
#importing the models



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
#shows the top five values of the data with thier attribute names.

train.head(8)

#prints all the attributes of the data.

print(train.columns.values)

#shows the bottom five values of the data with thier attribute names.

train.tail()

#gives the information abou the missing values in the data

train.info()

print("-"*40)

test.info()
#caluculates the mean std and all other things for the numerical valued attributes in the data

train.describe()
train.corr()
#gives the relation btw the attributes present in the data in the form of heatmap.

sns.heatmap(train.corr())
#plots a graph with the values present in the data

train.plot()
#gives the information abou the missing values in the data

train.info()
#prints all the values present in the given attribute.

train['Age']
#gives the information abou the missing values in the data

train.info()
#it gives the count, unique values, top values and freq of the data.

train.describe(include=['O'])
#caluculates the mean std and all other things for the numerical valued attributes in the data

train.describe()
#gives the relation btw the attributes with thier means with sorting by values.

train[['Pclass','Survived']].groupby(["Pclass"],as_index=False).mean().sort_values(by='Survived', ascending=False)

train[['Pclass','Survived']].groupby(['Pclass']).mean()
#gives the relation btw the attributes with thier means with sorting by values.

train[['Pclass','Survived']].groupby(["Pclass"],as_index=False).mean().sort_values(by='Survived', ascending=False)

#gives the relation btw the attributes with thier means with sorting by values.

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=True)
#gives the relation btw the attributes with thier means with sorting by values.

train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#gives the relation btw the attributes with thier means with sorting by values.

train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#gives a plot graph with respect to age and survived.

g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
#gives a plot graph of pclass, survived with respect to age and survived.

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
#gives a plot graph with respect to age and survived.

g = sns.FacetGrid(train, col='Survived', row='Pclass')

g.map(plt.hist, 'Age', bins=20)
#gives the relation btw fare and the age.

sns.jointplot(x='Fare',y='Age',data=train)
#it gives the missing values in the data

ms.matrix(train)
sns.distplot(train['Fare'])
train['Fare'].median()
sns.swarmplot(x='Pclass',y='Age',data=train,palette='Set2')
sns.set_style('whitegrid')

sns.countplot(x='Pclass',data=train)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
train.groupby('Pclass')['Age'].median()
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        # Class-1

        if Pclass == 1:

            return 37

        # Class-2 

        elif Pclass == 2:

            return 29

        # Class-3

        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
ms.matrix(train)
train.drop('Cabin', axis = 1,inplace=True)
train.head()
ms.matrix(train)
train.dropna(inplace=True)
ms.matrix(train)
train['Sex'].unique()
train['Sex'].value_counts()
sex_dummy = pd.get_dummies(train['Sex'],drop_first=True)

sex_dummy.head()
train['Embarked'].unique()
train['Embarked'].value_counts()
Embarked_dummy = pd.get_dummies(train['Embarked'],drop_first=True)

Embarked_dummy.head()
old_data = train.copy()

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.head()
old_data.info()
train.info
train = pd.concat([train,sex_dummy,Embarked_dummy],axis=1)
train.head()
train.info()
train.describe()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)

X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression



# Build the Model.

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predicted = logmodel.predict(X_test)
predicted.shape
889*0.3
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predicted))
from sklearn.metrics import precision_score



print(precision_score(y_test,predicted))
from sklearn.metrics import recall_score



print(recall_score(y_test,predicted))
from sklearn.metrics import f1_score

print(f1_score(y_test,predicted))
from sklearn.metrics import classification_report



print(classification_report(y_test,predicted))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predicted))
new = [[23,1,23.0,1,1,156.0009,0,0,0]]

logmodel.predict(new)

test.info()
test.head()
ms.matrix(test)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)



ms.matrix(test)



test.drop('Cabin', axis = 1, inplace= True)



ms.matrix(test)





test.fillna(test['Fare'].mean(),inplace=True)



test.info()



ms.matrix(test)



sex = pd.get_dummies(test['Sex'], drop_first=True)

embark = pd.get_dummies(test['Embarked'], drop_first=True)



test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)



test = pd.concat([test,sex,embark],axis=1)



test.head()
test.info()
predict1=logmodel.predict(test)



predict1



df1=pd.DataFrame(predict1,columns=['Survived'])



df2=pd.DataFrame(test['PassengerId'],columns=['PassengerId'])



df2.head()



result = pd.concat([df2,df1],axis=1)

result.head()

result.tail()
result.to_csv('result.csv',index=False)