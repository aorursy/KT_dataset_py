# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re as re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load data
train_file_path = '../input/train.csv'
test_file_path = '../input/test.csv'
test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)
complete_data = [train_data,test_data]
print(train_data.info())

print(train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean())
print(train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean())
for member in complete_data:
    member['FamilySize'] = member['SibSp'] + member['Parch'] + 1
print(train_data[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean())

for size in complete_data:
    size['IsAlone'] = 0
    size.loc[size['FamilySize'] == 1,'IsAlone'] = 1
print(train_data[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean())

for dataset in complete_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print(train_data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean())

for fare in complete_data:
    fare['Fare'] = fare['Fare'].fillna(fare['Fare'].median())
train_data['CategoricalFare'] = pd.qcut(train_data['Fare'],4)
print(train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'],as_index=False).mean())

for age in complete_data:
   average_age_train = age['Age'].mean()
   std_age_train = age['Age'].std()
   count_nan_age_train = age['Age'].isnull().sum()
   train_age_na_random = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
   age['Age'][np.isnan(age['Age'])] = train_age_na_random
   age['Age'] = age['Age'].astype(int)  
train_data['CategoricalAge'] = pd.cut(train_data['Age'],5)
print(train_data[['CategoricalAge','Survived']].groupby(['CategoricalAge'],as_index=False).mean())

def get_title_name(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

for name in complete_data:
    name['Title'] = name['Name'].apply(get_title_name)
    
print(pd.crosstab(train_data['Title'], train_data['Sex']))

for titles in complete_data:
    titles['Title'] = titles['Title'].replace(['Lady', 'Capt','Dona','Don', 'Jonkheer', 'Rev', 'Countess', 'Col', 'Dr', 'Sir', 'Major'], 'Rare')
    titles['Title'] = titles['Title'].replace('Mlle', 'Miss')
    titles['Title'] = titles['Title'].replace('Ms', 'Miss')
    titles['Title'] = titles['Title'].replace('Mme', 'Miss')
    
print(train_data[['Title','Survived']].groupby(['Title'],as_index=False).mean())





#Visulazing Data
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x='Embarked', y='Survived', hue ='Sex', data=train_data)
#sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=train_data)
#sns.barplot(x='Age', y='Survived', hue='Sex', data=train_data)
#train_data.Fare.describe()







#Data Cleaning
for dataset in complete_data:
    dataset['Sex'] = dataset['Sex'].map({'female':0,'male':1}).astype(int)
    
    title_dict = {'Master':1, 'Miss':2, 'Mr':3,'Mrs':4,'Rare':5}
    dataset['Title'] = dataset['Title'].map(title_dict).astype(int)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454),'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31),'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31,'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    
#drop_elements = ['PassengerId', 'Cabin', 'SibSp', 'Parch', 'Name', 'Ticket', 'FamilySize']
train_data = train_data.drop(['PassengerId', 'Cabin', 'SibSp', 'Parch', 'Name', 'Ticket', 'FamilySize'], axis = 1)
train_data = train_data.drop(['CategoricalAge','CategoricalFare'], axis = 1)
test_data  = test_data.drop(['Cabin', 'SibSp', 'Parch', 'Name', 'Ticket', 'FamilySize'], axis = 1)

#print(train_data.head(15))
#print(test_data.head())
#train_data = train_data.values
#test_data  = test_data.values    
    
X_train = train_data.drop("Survived",axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.drop('PassengerId',axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
sur_log = logreg.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
sur_rand = random_forest.score(X_train, Y_train)
svc =SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
sur_svc = svc.score(X_train, Y_train)
sur_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
sur_knn = knn.score(X_train, Y_train)
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
sur_gaussian = gaussian.score(X_train, Y_train)
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
sur_perceptron = perceptron.score(X_train, Y_train)
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
sur_linear = linear_svc.score(X_train, Y_train)
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sur_sgd = sgd.score(X_train, Y_train)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
print(Y_pred)
sur_decision_tree = decision_tree.score(X_train, Y_train)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [sur_svc, sur_knn, sur_log, 
              sur_rand, sur_gaussian, sur_perceptron, 
              sur_sgd, sur_linear, sur_decision_tree]})
models.sort_values(by='Score', ascending=False)
coeff_df = DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
coeff_df
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": Y_pred })
submission.to_csv('titanic.csv', index=False)