# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_pd = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train = train_pd.copy(deep = True)
data = [train,test]
train.info()

test.info()
train.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1,inplace = True)

test.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1,inplace = True)
for dataset in data:

   dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace = True)

   dataset['Fare'].fillna(dataset['Fare'].median(),inplace = True)

   

   
train.isnull().sum()
def impute_age(col):

    age=col[0]

    pclass = col[1]

    if pd.isnull(age):

        if pclass == 1:

         return 37

        elif pclass == 2:

         return 29

        else :

         return 24

    else :

     return age
for dataset in data:

   dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis = 1)
train.isnull().sum()
test.isnull().sum()
for dataset in data:

   dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1

   dataset['Isalone'] = 1

   dataset['Isalone'][dataset['Family']>1]=0
train['Fare'][train['Pclass']==3].median()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize = (18,20))

g = sns.FacetGrid(train,col = 'Survived',height = 5)

g= g.map(plt.hist,'Age',bins = 20)
def impute_age(col):

    age=col

    if age<=20:

         return 0

    elif age > 20 and age <=28:

         return 1

    elif age > 28 and age <38:

         return 2

    else : 

         return 3

for dataset in data :    

   dataset['Age'] = dataset['Age'].apply(impute_age)
def impute_age(col):

    age=col

    if age<=7.9:

         return 0

    elif age > 7.9 and age <=14.45:

         return 1

    elif age > 14.45 and age <31:

         return 2

    else : 

         return 3

for dataset in data :    

   dataset['Fare'] = dataset['Fare'].apply(impute_age)
g = sns.heatmap(train.corr(),annot = True)
for dataset in data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train
for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    
y_train = train['Survived']

train.drop('Survived',axis = 1 , inplace = True)
train
X_train, X_val, Y_train, Y_val = train_test_split(train, y_train, test_size = 0.3, random_state=1)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_val)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_val)

random_forest.score(X_train, Y_train)

print("Accuracy:",metrics.accuracy_score(Y_val, Y_pred))
Y_pred =logreg.predict(test)
Y_pred = pd.Series(Y_pred,name="Survived")

submission = pd.concat([pd.Series(range(892,1310),name = "PassengerId"),Y_pred],axis = 1)
submission.to_csv("titanic4.csv",index=False)
test.shape
Y_pred.shape