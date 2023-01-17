# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import the datasets

test_d = pd.read_csv("../input/test.csv")
train_d= pd.read_csv("../input/train.csv")

# Analyze the Datasets

train_d.describe()
train_d.info()

# Therefore, there are missing values in Age, Cabin and Embarked in the train dataset

test_d.describe()
test_d.info()

#Missing values in Age, Fare , Cabin
# 1) Compare Survived with different numerical data

# Comparing Pclass with Survived 

train_d[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)

# Comparing Sex with Survived 

train_d[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# Comparing SibSp with Survived

train_d[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# Comparing Parch with Survived

train_d[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)

# Visualizing

# Survived with Age
g = sns.FacetGrid(train_d, col = 'Survived')
g.map(plt.hist,'Age',bins=20)

# PClass with Age (Survived = 0 and 1)

g=sns.FacetGrid(train_d,col='Survived',row = 'Pclass')
g.map(plt.hist,'Age',bins=20)

# Sex and Survived

g = sns.FacetGrid(train_d,col='Survived')
g.map(plt.hist,'Sex',bins=10)


# Pclass, Sex and Embarked with Survived

g=sns.FacetGrid(train_d,row='Embarked')
g.map(sns.pointplot,'Pclass','Survived','Sex')
g.add_legend()

# Fare and Embarked with Survived

g=sns.FacetGrid(train_d,col='Survived',row='Embarked')
g.map(sns.barplot,'Sex','Fare')


"""CONCLUSIONS - 
 1) Pclass has direct correlation with Survived. Class 1 has highest survival rate followed by 2 followed by 1
 2) Sex has direct correlation with Survival rate. Female > Male.
 3) SibSp and Parch do not have a direct correlation, but we should make a new feature containing entir family size.
 4) Age and Survival rate have a correlation. Least number of deaths among children and old people. Should make age bands.
 5) Either embarked and survival has a correlation or embarked and Pclass. To determine what correlation embarked follows, we compare with Pclass and fare.
 6) High paying passengers are more likely to survive.Fare Bands must be made. Also, Embarked has a direct correlation with Survival.
 7) Embarked will have passengers of diff classes and genders. The location of the cabin etc. makes the correlation  
"""
"""
WRANGLE DATA

"""
""" Drop the Ticket, PassengerID,Cabin columns"""

train_d=train_d.drop(['Ticket','PassengerId','Cabin'],axis=1 )
test_d=test_d.drop(['Ticket','Cabin'],axis=1)

""" Combine SibSp and Parch to form FamilySize """

train_d['FamilySize']=train_d['Parch']+train_d['SibSp']+1
test_d['FamilySize']=test_d['Parch']+test_d['SibSp']+1

"""Drop Parch and SibSp"""

train_d= train_d.drop(['Parch','SibSp'],axis=1)
test_d=test_d.drop(['Parch','SibSp'],axis=1)

"""Filling Missing Values """
train_d.info()  # Age and Embarked Values Missing
test_d.info()   # Age and Fare value missing

"""FILLING MISSING VALUES FOR EMBARKED"""
mode_emb = train_d.Embarked.dropna().mode()[0]
train_d['Embarked']=train_d['Embarked'].fillna(mode_emb)


""" Label Encode Sex and Embarked"""

from sklearn.preprocessing import LabelEncoder
le_tr = LabelEncoder()
le1_tr= LabelEncoder()
le_test = LabelEncoder()
le1_test= LabelEncoder()
train_d['Sex']=le_tr.fit_transform(train_d['Sex'])
test_d['Sex']=le_test.fit_transform(test_d['Sex'])
train_d['Embarked']=le1_tr.fit_transform(train_d['Embarked'])
test_d['Embarked']=le1_test.fit_transform(test_d['Embarked'])

"""Extracting title from names"""
train_d['Title']=np.nan
for i in range(0,891):
    train_d['Title'][i]=(train_d['Name'][i].split(",")[1]).split(".")[0]

test_d['Title']=np.nan
for i in range(0,len(test_d)):
    test_d['Title'][i]=(test_d['Name'][i].split(",")[1]).split(".")[0]

#drop Name

train_d=train_d.drop('Name',axis=1)
test_d=test_d.drop('Name',axis=1)


train_d['Title']=train_d['Title'].replace([' Col',' Major',' Sir',' Don',' the Countess',' Jonkheer',' Capt'],' Rare')
train_d['Title']=train_d['Title'].replace([' Mlle',' Ms',' Mme',' Lady'],' Mrs')
train_d['Title'].value_counts()

test_d['Title']=test_d['Title'].replace([' Col',' Major',' Sir',' Don',' the Countess',' Jonkheer',' Capt',' Dona'],' Rare')
test_d['Title']=test_d['Title'].replace([' Mlle',' Ms',' Mme',' Lady'],' Mrs')

test_d['Title'].value_counts()

# Label Encoding Title
le_title=LabelEncoder()
train_d["Title"]=le_title.fit_transform(train_d["Title"])
test_d['Title']=le_title.transform(test_d['Title'])


# Filling fare values in test dataset- 

test_d.info()
test_d[test_d['Fare'].isnull()].index.tolist()

#Find avg of Fare of PClass = 3 

test_d['Fare'][152]=(test_d[test_d['Pclass']==3]['Fare'].mean())
test_d.loc[152,:]

test_d.info()
train_d.info()

#Fill missing age values based on name 
combine = [train_d,test_d]
titles = train_d['Title'].unique()
for dataset in combine:
    for t in titles:
        dataset['Age']=dataset['Age'].fillna(dataset[dataset['Title']==t]['Age'].mean()).astype(int)
    
train_d.info()
test_d.info()

#All values are filled. Let's create bands for age and Fare

train_d.describe()
train_d['AgeBand']=pd.cut(train_d['Age'],5)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
    
#Drop AgeBand
train_d=train_d.drop('AgeBand',axis=1)
combine=[train_d,test_d]

#Create fare bands
train_d['FareBands']=pd.qcut(train_d['Fare'],4)
train_d[['FareBands','Survived']].groupby(['FareBands'],as_index=False).mean().sort_values(by='FareBands')  
train_d['FareBands'].value_counts()
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    

train_d=train_d.drop('FareBands',axis=1)
train_d.describe()

"""
This completes the Cleaning of our data. Now we'll model it and find the best fitting model"""

x_train = train_d.drop('Survived',axis=1)
y_train= train_d['Survived']
x_test= test_d.drop("PassengerId",axis=1).copy()

""" Since it is a Classification and Regression problem (Supervised Learning), we import the relevant libraries"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

scorevector=[]

"""LOGISTIC REGRESSION"""
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred= lr.predict(x_test)
acc_log = round(lr.score(x_train, y_train) * 100, 2)
scorevector.append(acc_log)

"""SUPPPORT VECTOR CLASSIFICATION"""
sv = SVC()
sv.fit(x_train,y_train)
y_pred=sv.predict(x_test)
acc_log = round(sv.score(x_train, y_train) * 100, 2)
scorevector.append(acc_log)

"""GAUSSIAN NAIVE BAYES'"""
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred= gnb.predict(x_test)
acc_log = round(gnb.score(x_train, y_train) * 100, 2)
scorevector.append(acc_log)

"""RANDOM FOREST CLASSIFICATION"""
rfc = RandomForestClassifier(n_estimators=200,random_state=0)
rfc.fit(x_train,y_train)
y_pred= rfc.predict(x_test)
acc_log = round(rfc.score(x_train, y_train) * 100, 2)
scorevector.append(acc_log)

"""K-NEIGHBORS CLASSIFICATION"""
knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(x_train,y_train)
y_pred= knc.predict(x_test)
acc_log = round(knc.score(x_train, y_train) * 100, 2)
scorevector.append(acc_log)

"""DECISION TREE CLASSIFIER"""
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train,y_train)
y_pred= dtc.predict(x_test)
acc_log = round(dtc.score(x_train, y_train) * 100, 2)
scorevector.append(acc_log)


modelnames = pd.Series(['LogisticRegression','SVC','GaussianNB','RandomForestClassifier','KNeighborsClassifier','DecisionTreeClassifier'],dtype = object)
models = pd.DataFrame({'Model' : modelnames, 'Score':scorevector})
models.sort_values(by='Score',ascending=False)

submission = pd.DataFrame({"PassengerID" : test_d["PassengerId"], "Survived" : y_pred})
submission.to_csv('submission.csv', index=False)