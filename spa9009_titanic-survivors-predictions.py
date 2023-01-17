# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.shape
train.describe()
train.info()
test.shape

test.describe()
test.info()
survived = train[train["Survived"] == 1]
not_survived = train[train["Survived"] == 0]

print("Survived : %i (%.1f%%)"%(len(survived), (float)(len(survived)/len(train)*100)))

print("Not Survived : %i (%.1f%%)"%(len(not_survived), (float)(len(not_survived)/len(train)*100)))
print("Total : %i"%len(train))
train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()
sns.barplot(x = 'Pclass', y = 'Survived', data = train)
train.Sex.value_counts()
train.groupby('Sex').Survived.value_counts()
train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean()


sns.barplot(x = 'Sex', y = 'Survived', data = train)
tab = pd.crosstab(train['Pclass'],train['Sex'] )
print(tab)

tab.div(tab.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')
sns.factorplot('Sex', 'Survived', hue = 'Pclass', size = 4, aspect = True, data = train)
sns.factorplot('Sex', 'Survived', hue = 'Pclass',col = 'Embarked', size = 4, aspect = True, data = train)
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
sns.barplot(x = 'Embarked', y = 'Survived', data= train)
train.Parch.value_counts()
train.groupby('Parch').Survived.value_counts()

train[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean()

sns.barplot(x = 'Parch', y = 'Survived',ci = None, data= train)
fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

sns.violinplot(x = 'Pclass', y = 'Age',hue = 'Survived', ax = ax1, split = True, data = train)
sns.violinplot(x = 'Embarked', y = 'Age',hue = 'Survived', ax = ax2, split = True, data = train)
sns.violinplot(x = 'Sex', y = 'Age',hue = 'Survived', ax = ax3, split = True, data = train)

total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]

plt.figure(figsize=[15,5])
plt.subplot(111)
sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')

plt.figure(figsize=[15,5])

plt.subplot(121)
sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')

plt.subplot(122)
sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')
plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)
train_test_data = [train, test] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
train.head()
pd.crosstab(train['Title'], train['Sex'])
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Mme'],'Miss')
    dataset['Title'] = dataset['Title'].replace(['Ms'],'Mrs')
    
train.head()
pd.crosstab(train['Title'], train['Sex'])
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4,'Other':5}

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
train.head()

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':2}).astype(int)
train.head()
train.Embarked.unique()

train.Embarked.value_counts()
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head()
for dataset in train_test_data:
    #print(dataset.Embarked.unique())
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
train.head()
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

train.head()
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train.head()
for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
train.head()
train['Fareband'] = pd.qcut(train['Fare'], 4)
print(train[['Fareband', 'Survived']].groupby(train['Fareband']).mean())
train.head()
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
train.head(1)
test.head(1)
train.head()
test.head()
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId', 'AgeBand', 'Fareband'], axis=1)
train.head()

test.head()
X_train = train.drop('Survived', axis = 1)
y_train = train['Survived']
X_test = test.drop('PassengerId', axis = 1).copy()


X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_log_reg)+'%')

clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_svc)+'%')
    
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_lsvc = clf.predict(X_test)
acc_lsvc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_lsvc)+'%')
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_dtc = clf.predict(X_test)
acc_dtc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_dtc)+'%')
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred_knc = clf.predict(X_test)
acc_knc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_knc)+'%')
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred_rfc = clf.predict(X_test)
acc_rfc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_rfc)+'%')
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_gnb)+'%')
clf = Perceptron()
clf.fit(X_train, y_train)
y_pred_per = clf.predict(X_test)
acc_per = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_per)+'%')
clf = SGDClassifier()
clf.fit(X_train, y_train)
y_pred_sgdc = clf.predict(X_test)
acc_sgdc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_sgdc)+'%')
from sklearn.metrics import confusion_matrix
import itertools

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest_training_set = clf.predict(X_train)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy: %i %% \n"%acc_random_forest)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_random_forest_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_lsvc, 
              acc_knc,  acc_dtc, acc_rfc, acc_gnb, 
              acc_per, acc_sgdc]
    })

models.sort_values(by='Score', ascending=False)
test.head()
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_dtc
    })
submission.to_csv('submission.csv', index = False)
