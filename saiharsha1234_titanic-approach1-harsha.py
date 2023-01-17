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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a = pd.read_csv('../input/titanic/train.csv')
b = pd.read_csv('../input/titanic/test.csv')
a.info()
a.describe()
a_total_mean=a
b_total_mean=b
a_total_mean.shape,b_total_mean.shape
combine=[a,b]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(a['Title'], a['Sex'])
combine1=[a_total_mean,b_total_mean]
for dataset1 in combine1:
    dataset1['Title'] = dataset1.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(a_total_mean['Title'], a_total_mean['Sex'])
#method1
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
a[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#method2
for dataset1 in combine1:
    dataset1['Title'] = dataset1['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset1['Title'] = dataset1['Title'].replace('Mlle', 'Miss')
    dataset1['Title'] = dataset1['Title'].replace('Ms', 'Miss')
    dataset1['Title'] = dataset1['Title'].replace('Mme', 'Mrs')
    
a_total_mean[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
a[['Title', 'Survived']].groupby(['Title'], as_index=False).count()
a[['Title', 'Survived']].groupby(['Title'], as_index=False).sum()
T=a['Title'].unique()
res=[]
for Tech in T:
    tmp=a[a['Title']==Tech]
    tmp['Age']=tmp.Age.fillna(tmp.Age.mean())
    res.append(tmp)

    
a1=pd.concat(res,sort=True)    
a1.info()
res1=[]
for Tech1 in T:
    tmp1=b[b['Title']==Tech1]
    tmp1['Age']=tmp1.Age.fillna(tmp1.Age.mean())
    res1.append(tmp1)
b1=pd.concat(res1,sort=True)
b1.info()
a_total_mean['Age']=a_total_mean.Age.fillna(a_total_mean.Age.mean())
b_total_mean['Age']=b_total_mean.Age.fillna(b_total_mean.Age.mean())
a_total_mean.info()
b_total_mean.info()
#METHOD-1
combine=[a1,b1]
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

a1.head()
combine1=[a_total_mean,b_total_mean]
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset1 in combine1:
    dataset1['Title'] = dataset1['Title'].map(title_mapping)
    dataset1['Title'] = dataset1['Title'].fillna(0)

a_total_mean.head()
#METHOD-1 
a1 = a1.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)
b1 = b1.drop(['Ticket', 'Cabin','Name'], axis=1)
#METHOD-2
a_total_mean = a_total_mean.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)
b_total_mean = b_total_mean.drop(['Ticket', 'Cabin','Name'], axis=1)
#METHOD-1
b1['Fare']=b1.Fare.fillna(b1.Fare.mean())
#METHOD-2
b_total_mean['Fare']=b_total_mean.Fare.fillna(b_total_mean.Fare.mean())
a1.info()
#METHOD-1
combine=[a1,b1]
for dataset in combine:
    dataset['Age']=dataset['Age'].astype(int)
    dataset['Fare']=dataset['Fare'].astype(int)
#METHOD-2
combine1=[a_total_mean,b_total_mean]
for dataset in combine1:
    dataset['Age']=dataset['Age'].astype(int)
    dataset['Fare']=dataset['Fare'].astype(int)
a['Embarked'].value_counts()
import seaborn as sns
sns.countplot(x='Embarked',data=a)
plt.show()
#METHOD-1
common_value = 'S'
combine = [a1,b1]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
#METHOD-2
common_value = 'S'
combine1 = [a_total_mean,b_total_mean]

for dataset1 in combine1:
    dataset1['Embarked'] = dataset1['Embarked'].fillna(common_value)
#METHOD-1
combine = [a1, b1]

for dataset in combine:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
#METHOD-2
combine1 = [a_total_mean, b_total_mean]

for dataset1 in combine1:
    dataset1['Fare'] = dataset1['Fare'].fillna(0)
    dataset1['Fare'] = dataset1['Fare'].astype(int)
#METHOD-1
combine = [a1, b1]
for dataset in combine:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
#METHOD-2
combine1 = [a_total_mean, b_total_mean]
for dataset1 in combine1:
    dataset1['relatives'] = dataset1['SibSp'] + dataset1['Parch']
    dataset1.loc[dataset1['relatives'] > 0, 'not_alone'] = 0
    dataset1.loc[dataset1['relatives'] == 0, 'not_alone'] = 1
    dataset1['not_alone'] = dataset1['not_alone'].astype(int)
#METHOD-1
ports = {"S": 0, "C": 1, "Q": 2}
combine = [a1, b1]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
#METHOD-2
ports = {"S": 0, "C": 1, "Q": 2}
combine1 = [a_total_mean, b_total_mean]

for dataset1 in combine1:
    dataset1['Embarked'] = dataset1['Embarked'].map(ports)
#METHOD-1
genders = {"male": 0, "female": 1}
combine = [a1, b1]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(genders)
#METHOD-2
genders = {"male": 0, "female": 1}
combine1 = [a_total_mean, b_total_mean]

for dataset1 in combine1:
    dataset1['Sex'] = dataset1['Sex'].map(genders)
#METHOD-1
a2=a1.drop(['Survived','Parch','SibSp'],axis=1)
b1=b1.drop(['PassengerId','Parch','SibSp'],axis=1)
a2.shape,b1.shape
#METHOD-2
a2_totalmean=a_total_mean.drop(['Survived','Parch','SibSp'],axis=1)
b1_totalmean=b_total_mean.drop(['PassengerId','Parch','SibSp'],axis=1)
a2_totalmean.shape,b1_totalmean.shape
a_array=a2.iloc[:,0:8].values
b_array=b1.iloc[:,0:8].values
a_array
a_array_totalmean=a2_totalmean.iloc[:,0:8].values
b_array_totalmean=b1_totalmean.iloc[:,0:8].values
a_array_totalmean
from sklearn.preprocessing import OneHotEncoder
one_array=OneHotEncoder()
one_array_totalmean=OneHotEncoder()
cols=[1,3,5]
col=[0,4,5]   #position are different so that's why we are taking different numbers
one_array.fit(a_array[:,cols])
one_array_totalmean.fit(a_array_totalmean[:,col])
a2_encoded=one_array.transform(a_array[:,cols])
a2_encoded_totalmean=one_array_totalmean.transform(a_array_totalmean[:,col])
b1_encoded=one_array.transform(b_array[:,cols])
b1_encoded_totalmean=one_array_totalmean.transform(b_array_totalmean[:,col])
a2_encoded.shape
a2_encoded_totalmean.shape
b1_encoded.shape
b1_encoded_totalmean.shape
#METHOD-1
X_train = a2_encoded
Y_train = a1["Survived"]
X_test  = b1_encoded
#METHOD-2
X_train_totalmean = a2_encoded_totalmean
Y_train_totalmean = a_total_mean["Survived"]
X_test_totalmean  = b1_encoded_totalmean
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
#METHOD-1
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3,random_state=1,shuffle=True)
#METHOD-2
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train_totalmean,Y_train_totalmean,test_size=0.3,random_state=1,shuffle=True)
from sklearn.metrics import accuracy_score
#METHOD-1
# stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)    #number of times trained in dataset
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_test, Y_pred) * 100, 2)

print(round(acc_sgd,2,), "%")
#METHOD-2
# stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)    #number of times trained in dataset
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_test, Y_pred) * 100, 2)      #total age mean

print(round(acc_sgd,2,), "%")
#METHOD-1
sgd.score(x_test,y_test)
#METHOD-2
sgd.score(x_test,y_test)   #total age mean
#METHOD-1
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

Y_prediction = random_forest.predict(x_test)


acc_random_forest = round(accuracy_score(y_test,Y_prediction) * 100, 2)
print(round(acc_random_forest,2,), "%")
#METHOD-2
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

Y_prediction = random_forest.predict(x_test)                     #total age mean


acc_random_forest = round(accuracy_score(y_test,Y_prediction) * 100, 2)
print(round(acc_random_forest,2,), "%")
#METHOD-1
random_forest.score(x_test,y_test)
#METHOD-2
random_forest.score(x_test,y_test)   #total age mean
#METHOD-1
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

Y_pred = logreg.predict(x_test)

acc_log = round(accuracy_score(y_test, Y_pred) * 100, 2)
print(round(acc_log,2,), "%")
#METHOD-2
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)            #total age mean

Y_pred = logreg.predict(x_test)

acc_log = round(accuracy_score(y_test, Y_pred) * 100, 2)
print(round(acc_log,2,), "%")
#METHOD-1
logreg.score(x_test,y_test)
#METHOD-2
logreg.score(x_test,y_test)       #total age mean
#METHOD-1
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_knn,2,), "%")
#METHOD-2
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)                      #total age mean

y_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_knn,2,), "%")
#METHOD-1
knn.score(x_test,y_test)
#METHOD-2
knn.score(x_test,y_test)   #total age mean
#METHOD-1
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train.toarray(), y_train)

y_pred = gaussian.predict(x_test.toarray())

acc_gaussian = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_gaussian,2,), "%")
#METHOD-2
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train.toarray(), y_train)    #total age mean

y_pred = gaussian.predict(x_test.toarray())

acc_gaussian = round(accuracy_score(y_test,y_pred) * 100, 2)
print(round(acc_gaussian,2,), "%")
#METHOD-1
gaussian.score(x_test.toarray(),y_test)
#METHOD-2
gaussian.score(x_test.toarray(),y_test)   #total age mean
#METHOD-1
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)

Y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_linear_svc,2,), "%")
#METHOD-2
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)   #total age mean

Y_pred = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_linear_svc,2,), "%")
#METHOD-1
linear_svc.score(x_test,y_test)
#METHOD-2
linear_svc.score(x_test,y_test)   #total age mean
#METHOD-1
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_decision_tree,2,), "%")
#METHOD-2
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)       #total age mean

Y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(accuracy_score(y_test,Y_pred) * 100, 2)
print(round(acc_decision_tree,2,), "%")
#METHOD-1
decision_tree.score(x_test,y_test)
#METHOD-2
decision_tree.score(x_test,y_test)    #total age mean
#METHOD-1

results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',  
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log,                #in this cells the score are (x_test,y_test)
              acc_random_forest, acc_gaussian, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
#METHOD-2

results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',  
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log,                #in this cells the score are (x_test,y_test)
              acc_random_forest, acc_gaussian, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)     #total age mean
result_df = result_df.set_index('Score')
result_df.head(9)
