# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a=pd.read_csv('../input/titanic/train.csv')
b=pd.read_csv('../input/titanic/test.csv')

a.info()
a.describe()
combine=[a,b]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(a['Title'], a['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
a[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
a[['Title', 'Survived']].groupby(['Title'], as_index=False).count()
a[['Title', 'Survived']].groupby(['Title'], as_index=False).sum()
T=a['Title'].unique()
res=[]
for Tech in T:
    tmp=a[a['Title']==Tech]
    tmp['Age']=tmp.Age.fillna(tmp.Age.mean())
    res.append(tmp)

    
a1=pd.concat(res,sort=True)    
res1=[]
for Tech1 in T:
    tmp1=b[b['Title']==Tech1]
    tmp1['Age']=tmp1.Age.fillna(tmp1.Age.mean())
    res1.append(tmp1)
b1=pd.concat(res1,sort=True)
combine=[a1,b1]
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

a1.head()
a1.info()
b1.info()
a1 = a1.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)
b1 = b1.drop(['Ticket', 'Cabin','Name'], axis=1)
a1.shape,b1.shape
b1['Fare']=b1.Fare.fillna(b1.Fare.mean())
combine=[a1,b1]
for dataset in combine:
    dataset['Age']=dataset['Age'].astype(int)
    dataset['Fare']=dataset['Fare'].astype(int)
a['Embarked'].value_counts()
import seaborn as sns
sns.countplot(x='Embarked',data=a)
plt.show()
common_value = 'S'
combine = [a1,b1]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
combine = [a1, b1]

for dataset in combine:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
ports = {"S": 0, "C": 1, "Q": 2}
combine = [a1, b1]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
genders = {"male": 0, "female": 1}
combine = [a1, b1]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(genders)
a1.info()
b1.info()
a1.to_csv('train_processed_continous.csv')
a1.head()
b1.to_csv('test_processed_continous.csv')
b1.head()
a2=a1.drop('Survived',axis=1)
b1=b1.drop('PassengerId',axis=1)
a2.shape,b1.shape
a_array=a2.iloc[:,0:8].values
b_array=b1.iloc[:,0:8].values
a_array
b_array
from sklearn.preprocessing import OneHotEncoder
one_array=OneHotEncoder()
cols=[1,4,7]
one_array.fit(a_array[:,cols])
a2_encoded=one_array.transform(a_array[:,cols])
b1_encoded=one_array.transform(b_array[:,cols])
a2_encoded.shape
b1_encoded.shape
X_train = a2_encoded
Y_train = a1["Survived"]
X_test  = b1_encoded
# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3,random_state=1,shuffle=True)
from sklearn.metrics import accuracy_score
# stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)    #number of times trained in dataset
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_test, Y_pred) * 100, 2)

print(round(acc_sgd,2,), "%")
sgd.score(x_test,y_test)
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

Y_prediction = random_forest.predict(x_test)

acc_random_forest = round(accuracy_score(y_test,Y_prediction) * 100, 2)
print(round(acc_random_forest,2,), "%")
random_forest.score(x_test,y_test)
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

Y_pred = logreg.predict(x_test)

acc_log = round(accuracy_score(y_test, Y_pred) * 100, 2)
print(round(acc_log,2,), "%")
logreg.score(x_test,y_test)
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_knn,2,), "%")
knn.score(x_test,y_test)
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train.toarray(), y_train)

y_pred = gaussian.predict(x_test.toarray())

acc_gaussian = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_gaussian,2,), "%")
gaussian.score(x_test.toarray(),y_test)
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)

Y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_linear_svc,2,), "%")
linear_svc.score(x_test,y_test)
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_decision_tree,2,), "%")
decision_tree.score(x_test,y_test)
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',  
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log,                #in this cells the score are (x_test,y_test)
              acc_random_forest, acc_gaussian,                 #Parch,Sibsp,Age,Fare are continuos
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
