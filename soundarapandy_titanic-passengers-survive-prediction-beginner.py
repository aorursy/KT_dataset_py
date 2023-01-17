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
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import f1_score

from sklearn import metrics

import seaborn as sns
train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")

print(train_df.info())

print(test_df.info())
print(train_df.describe())

print('-'*50)

print(test_df.describe())
print(train_df.isnull().sum())

print(test_df.isnull().sum())
#To standardize the nan columns in Age feature by replace it by simple arthimetic mean

train_df.replace(to_replace=np.nan,value=round(train_df.mean(),0),inplace=True)

train_df.fillna(method='pad',inplace=True)

test_df.replace(to_replace=np.nan,value=round(test_df.mean(),0),inplace=True)

test_df.fillna(method='pad',inplace=True)

print(train_df.info())

print('_'*50)

print(test_df.info())
print(train_df.columns.values)

print(test_df.columns.values)
print(train_df.shape)

print(test_df.shape)
train_df = train_df.replace({'male':0,'female':1})

test_df = test_df.replace({'male':0,'female':1})

print(train_df.head(1))

print(test_df.head(1))
#Survived— Whether the passenger survived or not and the value we are predicting (0=No, 1=Yes)

#Pclass— The class of the ticket the passenger purchased (1=1st, 2=2nd, 3=3rd)

#Sex— The passenger’s sex

#Age— The passenger’s age in years

#SibSp— The number of siblings or spouses the passenger had aboard the Titanic

#Parch— The number of parents or children the passenger had aboard the Titanic

#Ticket— The passenger’s ticket number

#Fare— The fare the passenger paid

#Cabin— The passenger’s cabin number

#Embarked— The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)
#Lets drop the unwanted columns

#since cabin contains maximum of Nan columns so that can be drop 

remove_unwanted_columns = ['Cabin','Ticket','Name','PassengerId','Embarked']

train_df = train_df.drop(remove_unwanted_columns,axis=1)

test_df = test_df.drop(remove_unwanted_columns,axis=1)

print(train_df.columns.values)

print(test_df.columns.values)
print(train_df.shape)

print(test_df.shape)
#view of mean 'survived' people on compare with its respected features(Pclass,Sex,SibSp,Parch,Age)

print('Pclass:',train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))

print('_'*40)

print('Sex:',train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False))

print('_'*40)

print('SibSp:',train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False))

print('_'*40)

print('Parch:',train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False))
#view of correaltion between features

feature_selection = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']

corr = train_df[feature_selection].corr()

sns.heatmap(corr,annot=True,annot_kws={'size':9},linewidths=0.5,xticklabels=feature_selection,

           yticklabels=feature_selection,cmap='coolwarm')

#barplot

#Sex vs Survived

pal = {0:'green',1:'red'}

sns.set(style="darkgrid")

plt.subplots(figsize=(15,8))

ax = sns.barplot(x='Sex',y='Survived',data=train_df,palette=pal,linewidth=5,order=[0,1],capsize=0.5)

plt.title('Sex/Survived',loc='center',pad=40)

plt.ylabel('% of passenger Survived')

plt.xlabel('Sex')
#Pclass vs Survived

plt.subplots(figsize=(15,10))

ax = sns.barplot(x='Pclass',y='Survived',data=train_df,linewidth=5,capsize=0.1)

plt.title('Pclass/Sex',pad=40,loc='center')

plt.ylabel('% of passengers Survived')

plt.xlabel('Pclass')

#SibSP vs Survived

sns.set(style='darkgrid')

plt.subplots(figsize=(10,15))

ax = sns.barplot(x='SibSp',y='Survived',data=train_df,linewidth=5,capsize=0.1)

plt.title('SibSp/Survived',loc='center',pad=40)

plt.ylabel('% of passengers survived')

plt.xlabel('SibSP')
#Parch vs Survived

sns.set(style='darkgrid')

plt.subplots(figsize=(15,8))

ax = sns.barplot(x='Parch',y='Survived',data=train_df,linewidth=5,capsize=.05)

plt.title('Parch/Survived',loc='center',pad=40)

plt.ylabel('% of passengers survived')

plt.xlabel('Parch')
#Sex vs Survived

a = train_df['Sex']

b = train_df['Survived']

sns.countplot(a,label='count')

plt.title('Sex/Survived',loc='center',pad=40)

plt.ylabel('passengers Survived')

plt.xlabel('Sex')
#Pclass vs Survived

a = train_df['Pclass']

b = train_df['Survived']

sns.countplot(a,label='count')

plt.title('Pclass/Survived',loc='center',pad=40)

plt.ylabel('passengers Survived')

plt.xlabel('Pclass')
#SibSp

a = train_df['SibSp']

b = train_df['Survived']

sns.countplot(a,label='count')

plt.title('SibSP/Survived')

plt.ylabel('passengers survived')

plt.xlabel('SibSp')

#Parch vs Survived

a = train_df['Parch']

b = train_df['Survived']

sns.countplot(a,label='count')

plt.title('Parch/Survived',loc='center',pad=40)

plt.ylabel('passengers survived')

plt.xlabel('Parch')
#outlier Detection

sns.set_style('darkgrid')

fig,ax = plt.subplots(figsize=(16,12),ncols=2)

ax1 = sns.boxplot(x='Parch',y='Fare',hue='Pclass',data=train_df,ax=ax[0]);

ax2 = sns.boxplot(x='Parch',y='Fare',hue='Pclass',data=test_df,ax=ax[1]);

ax1.set_title('Training_data',fontsize=18)

ax2.set_title('Testing_data',fontsize=18)
#let

X_train = train_df.iloc[:,1:7]

Y_train = train_df.iloc[:,0]



X_test = test_df.iloc[:,]



Y_test = pd.read_csv('../input/titanic/gender_submission.csv')
#MinMaxscaler to normalize the value between 0 and 1

scale = MinMaxScaler(feature_range=(0,1))

scaled_train = scale.fit_transform(X_train)

print('scaled_train:',scaled_train)

print('_'*90)

#Do similar for Test set

scale = MinMaxScaler(feature_range=(0,1))

scaled_test =  scale.fit_transform(X_test)

print('scaled_test:',scaled_test)
#Now implement the Machine Learning Algorithm ot predict the values and find accuracy

#LogisticRegression

lo_reg = LogisticRegression()

lo_reg.fit(scaled_train,Y_train)

Y_pred = lo_reg.predict(X_test)

#print(Y_pred)

Acc_lo_reg = metrics.accuracy_score(Y_test['Survived'], Y_pred) * 100

print(Acc_lo_reg)

print('*'*40)

#Confusion matrix

print(classification_report(Y_test['Survived'],Y_pred))

cnf_matrix = metrics.confusion_matrix(Y_test['Survived'],Y_pred)

print('cnf_matrix:',cnf_matrix)

sns.heatmap(cnf_matrix,annot=True,fmt='d')

#DecisionTreeClassifier

clf_tree = DecisionTreeClassifier()

clf_tree.fit(scaled_train,Y_train)

Y_pred = clf_tree.predict(X_test)

Acc_clf_tree = accuracy_score(Y_test['Survived'],Y_pred)*100

print(Acc_clf_tree)

print('*'*40)

print(classification_report(Y_test['Survived'],Y_pred))

cnf_matrix = metrics.confusion_matrix(Y_test['Survived'],Y_pred)

print('cnf_matrix:',cnf_matrix)

sns.heatmap(cnf_matrix,annot=True,fmt='d')
#KNeighborsclassifier

clf_knn = KNeighborsClassifier()

clf_knn.fit(scaled_train,Y_train)

Y_pred = clf_knn.predict(X_test)

Acc_clf_knn = accuracy_score(Y_test['Survived'],Y_pred)*100

print(Acc_clf_knn)

print('*'*40)

print(classification_report(Y_test['Survived'],Y_pred))

cnf_matrix = metrics.confusion_matrix(Y_test['Survived'],Y_pred)

print('cnf_matrix:',cnf_matrix)

sns.heatmap(cnf_matrix,annot=True,fmt='d')
#RandomForestClassfier

clf_random = RandomForestClassifier()

clf_random.fit(scaled_train,Y_train)

Y_pred = clf_random.predict(X_test)

Acc_clf_random = accuracy_score(Y_test['Survived'],Y_pred)*100

print(Acc_clf_random)

print('*'*40)

print(classification_report(Y_test['Survived'],Y_pred))

cnf_matrix = metrics.confusion_matrix(Y_test['Survived'],Y_pred)

print('cnf_matrix:',cnf_matrix)

sns.heatmap(cnf_matrix,annot=True,fmt='d')
#Support vector machine

clf_svm = RandomForestClassifier()

clf_svm.fit(scaled_train,Y_train)

Y_pred = clf_svm.predict(X_test)

Acc_clf_svm = accuracy_score(Y_test['Survived'],Y_pred)*100

print(Acc_clf_svm)

print('*'*40)

print(classification_report(Y_test['Survived'],Y_pred))

cnf_matrix = metrics.confusion_matrix(Y_test['Survived'],Y_pred)

print('cnf_matrix:',cnf_matrix)

sns.heatmap(cnf_matrix,annot=True,fmt='d')
#GaussianNB

clf_naive = GaussianNB()

clf_naive.fit(scaled_train,Y_train)

Y_pred = clf_naive.predict(X_test)

Acc_clf_naive = accuracy_score(Y_test['Survived'],Y_pred)*100

print(Acc_clf_naive)

print('*'*40)

print(classification_report(Y_test['Survived'],Y_pred))

cnf_matrix = metrics.confusion_matrix(Y_test['Survived'],Y_pred)

print('cnf_matrix:',cnf_matrix)

sns.heatmap(cnf_matrix,annot=True,fmt='d')
#SGDClassifier

clf_SGD = SGDClassifier()

clf_SGD.fit(scaled_train,Y_train)

Y_pred = clf_SGD.predict(X_test)

Acc_clf_SGD = accuracy_score(Y_test['Survived'],Y_pred)*100

print(Acc_clf_SGD)

print('*'*40)

print(classification_report(Y_test['Survived'],Y_pred))

cnf_matrix = metrics.confusion_matrix(Y_test['Survived'],Y_pred)

print('cnf_matrix:',cnf_matrix)

sns.heatmap(cnf_matrix,annot=True,fmt='d')
models = pd.DataFrame({

    'Model': ['lo_reg','clf_tree','clf_knn','clf_naive','clf_svm','clf_random','clf_SGD'],

    'Score': [Acc_lo_reg,Acc_clf_tree,Acc_clf_knn,Acc_clf_naive,Acc_clf_svm,

              Acc_clf_random,Acc_clf_SGD]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": Y_test["PassengerId"],

        "Survived": Y_pred

    })

print(submission)