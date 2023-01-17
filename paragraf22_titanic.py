import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualisation

import seaborn as sns # visualisation

#misc libraries

import random

import time

#ignore warnings

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.sample(10)
train.info()
print("Train data with null values: ",train.isnull().sum())
print('Test data with null values:\n', test.isnull().sum())
train.describe(include='all')
test_index = test['PassengerId']
plt.figure(figsize=(8,6))

fig_surv =  train.Survived.value_counts(normalize=True).plot(kind="bar",color=['black','green'],width=0.3)

fig_surv.set_title("Percent of people who have survived")

fig_surv.set_xticklabels(('Dead', 'Survied'))
plt.figure(figsize=(8, 6))

fig_sex =  train.Sex.value_counts(normalize=True).plot(kind="pie",autopct='%1.1f%%',shadow=True,colors=['blue','pink'])

fig_sex.set_title("Sex of passengers")

fig_sex.set_xticklabels(('Male', 'Female'))
plt.figure(figsize=(8, 6))

fig_age_far = train.plot(kind='scatter',x='Age',y='Fare',alpha=0.5,color='green')

fig_age_far.set_title("Correlation between age and fare")

plt.show()
plt.figure(figsize=(8, 6))

fig_box= train.boxplot(column='Age',by='Survived')

plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)

train.plot(kind = "hist",y = "Age",bins = 25,range= (0,85),normed = True,ax = axes[0])

train.plot(kind = "hist",y = "Age",bins = 25,range= (0,85),normed = True,ax = axes[1],cumulative = True)

plt.show()
plt.figure(figsize=(8, 6))

fig_class =  train.Pclass.value_counts(normalize=True).plot(kind="bar",color=['brown','gold','silver'],width=0.3)

fig_class.set_title("Class of cabins")

fig_class.set_xticklabels(('3rd', '1st','2st'))
plt.figure(figsize=(12, 8))

for x in [1,2,3]:

    train.Age[train.Pclass==x].plot(kind='kde')

plt.title("Class wrt Age")

plt.legend(['1st','2nd','3rd'])
plt.figure(figsize=(12, 8))

sns.heatmap(train.corr(),annot=True, linewidths=.5)
full_data = [train,test]

for dataset in full_data:

    dataset['Sex']=dataset['Sex'].map({'male':1,'female':0}).astype(int)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)
med_age = dataset['Age'].median()

std_age = dataset['Age'].std()



for dataset in full_data:

    dataset.loc[dataset['Age'].isnull(), 'Age']=random.randint(int(med_age-std_age), int(med_age+std_age))

    dataset['Age'] = dataset['Age'].astype(int)
drop_columns = ['PassengerId','Cabin', 'Ticket']

train.drop(drop_columns,axis=1,inplace=True)

test.drop(drop_columns,axis=1,inplace=True)
for dataset in full_data:

    dataset['Title'] = dataset['Name'].str.split(",",expand=True)[1].str.split('.',expand=True)[0]

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
title_names1 = (train['Title'].value_counts() < 10)

title_names2 = (test['Title'].value_counts() < 10)

train['Title']=train['Title'].apply(lambda x: "Misc" if title_names1[x]==True else x)

test['Title']=test['Title'].apply(lambda x: "Misc" if title_names2[x]==True else x)
train.sample(10)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data_x = ['Pclass', 'Embarked', 'Title','FareBin','AgeBin','SibSp','Parch','Sex'] 

train_dummy = pd.get_dummies(train[data_x])

test_dummy = pd.get_dummies(test[data_x])

columns_list = train_dummy.columns
target = train['Survived']

train_dummy  =  pd.concat([train_dummy, target], axis=1, join='inner')
from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(train_dummy[columns_list], target, random_state = 42,test_size=0.3)
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(train1_x_dummy, train1_y_dummy)

gnb_pred = gnb.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,gnb_pred)
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)

from sklearn.pipeline import Pipeline

nca = NeighborhoodComponentsAnalysis(random_state=42)

knn = KNeighborsClassifier(n_neighbors=12)

nca_pipe = Pipeline([('nca', nca), ('knn', knn)])

nca_pipe.fit(train1_x_dummy, train1_y_dummy) 

nca_pred = nca_pipe.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,nca_pred)
from sklearn import svm

svm_clf = svm.SVC(gamma='scale',C=1,cache_size=100,kernel='poly',coef0=0.5)

svm_clf.fit(train1_x_dummy, train1_y_dummy)

svm_pred = svm_clf.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,svm_pred)
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier(max_depth=100 ,min_samples_split=4,min_samples_leaf=6)

tree_clf.fit(train1_x_dummy, train1_y_dummy)

tree_pred = tree_clf.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,tree_pred)
from sklearn.ensemble import RandomForestClassifier

random_clf = RandomForestClassifier(n_estimators=500,max_depth=100,random_state=42,min_samples_split=4,min_samples_leaf=6)

random_clf.fit(train1_x_dummy, train1_y_dummy)

random_pred = random_clf.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,random_pred)
from sklearn.linear_model import SGDClassifier

SGD_clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

SGD_clf.fit(train1_x_dummy, train1_y_dummy)  

SGD_pred = SGD_clf.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,SGD_pred)

from sklearn.ensemble import GradientBoostingClassifier

GDC_clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.75,max_depth=2, random_state=42)

GDC_clf.fit(train1_x_dummy, train1_y_dummy)

GDC_pred = GDC_clf.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,GDC_pred)

from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,solver='lbfgs', multi_class='auto')

log_clf.fit(train1_x_dummy, train1_y_dummy)

log_pred = log_clf.predict(test1_x_dummy)

accuracy_score(test1_y_dummy,log_pred)
algorithm_dist = {'LogisticRegression':log_clf,'GradientBoosting':GDC_clf,'SGDClassifier':SGD_clf,'RandomTreeClassifier':random_clf,'DecisionTreeClassifier':tree_clf,'SVM':svm_clf,'KNeighborsClassifier':nca_pipe,'GaussianNB':gnb}
from sklearn.model_selection import cross_validate

for key,value in algorithm_dist.items():

    results = cross_validate(value, train1_x_dummy, train1_y_dummy, cv=10, scoring='accuracy',return_train_score=True, return_estimator=False, n_jobs=-1)

    print(key + ' train score:', results['train_score'].mean())

    print(key +' test score:', results['test_score'].mean())
final_pred = svm_clf.predict(test_dummy)

output = pd.DataFrame({'PassengerId': test_index, 'Survived': final_pred.round().astype(int)})

output.to_csv("submission.csv", index = False)