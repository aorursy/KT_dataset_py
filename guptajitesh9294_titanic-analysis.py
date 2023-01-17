# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing Libraries

# Data Analysis and wrangling
import numpy as np
import pandas as pd
import random as rnd

# Visualization
import  matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#MAchine learning algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

train =pd.read_csv("..//input//train.csv")
test=pd.read_csv("..//input//test.csv")
combine=[train,test]
train.head()
train.info()
# Numerical Variable
num_train=train.select_dtypes(include=np.number)
print(num_train.columns)
#Categorical Data
cat_train=train.select_dtypes(exclude=np.number)
print(cat_train.columns)
train.describe()
# age is lie between 0.4 to 80
## Now check the missing Data in training set
miss1=train.isnull().sum().sort_values(ascending=False)
miss= (train.isnull().sum()/len(train))*100
miss_data=pd.concat([miss1,miss],axis=1,keys=['Total','%'])
print(miss_data)
print("Age ,Cabin,Embarked has missing values")
## Now check the missing Data in test set
miss1=test.isnull().sum().sort_values(ascending=False)
miss= (test.isnull().sum()/len(test))*100
miss_data=pd.concat([miss1,miss],axis=1,keys=['Total','%'])
print(miss_data)
print("Age ,Cabin,Fare has missing values")
train.info()
print("--"*40)
test.info()
# Describing Cat training set
cat_train.columns
#check correlation
corr_data=train.corr()
sns.heatmap(corr_data,annot=True)
# NOw pivot and made some inferences from them
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by='Survived',ascending=False)
# 1.Age Vs Survived
g=sns.FacetGrid(train,col='Survived')
g.map(plt.hist,'Age',bins=20)
# 2. Pclass Vs Survived
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
survived='survived'
not_survived='not survived'
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,4))
women=train[train['Sex']=='female']
men=train[train['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(),bins=40,label= survived, ax = axes[0], kde = False)
ax.legend()
ax.set_title('Female')
ax=sns.distplot(men[men['Survived']==1].Age.dropna(),bins=18,label=not_survived,ax=axes[1],kde=False)
ax=sns.distplot(men[men['Survived']==0].Age.dropna(),bins=40,label=not_survived,ax=axes[1],kde=False)
ax.legend()
ax.set_title('Male')
# You can see that men has high probability between Age 18 to 35 while women has survival rate between
# age 14 to 40
# for men the probability rate between 0 to 10 is high but for isnt true for female Another thing to note 
# is that infants also have a little bit higher probability of survival.
FacetGrid=sns.FacetGrid(train,row='Embarked',size=4.5 , aspect= 1.6)
FacetGrid.map(sns.pointplot,'Pclass','Survived','Sex')
FacetGrid.add_legend()
# Embarked seems to be correlated with survival, depending on the gender.

# Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port
# C. Men have a high survival probability if they are on port C, but a low probability if they are on port
# Q or S.

# Pclass also seems to be correlated with survival. We will generate another plot of it below.
sns.barplot(x='Pclass',y='Survived', data=train)
# Here we see clearly, that Pclass is contributing to a persons chance of survival, especially if this
# person is in class 1. We will create another pclass plot below.
grid=sns.FacetGrid(train,col='Survived',row='Pclass')
grid.map(plt.hist,'Age')
grid.add_legend()
# this plot confirms our assumption that pclass 1 has high Survival rate while it also shows that most of
# the people died in Pclass 3
# This combination make a sense because that total number of relatives a person has,on Titanic
data=[train,test]
for dataset in data:
    dataset['Relatives']=dataset['SibSp']+dataset['Parch']
    dataset.loc[dataset['Relatives']>0,'not_alone']=0
    dataset.loc[dataset['Relatives']==0,'not_alone']=1
    dataset['not_alone']=dataset['not_alone'].astype(int)
train['not_alone'].value_counts()
axes=sns.factorplot('Relatives','Survived',data=train)
# Here we can see that you had a high probabilty of survival with 1 to 3 realitves, but a lower one if
# you had less than 1 or more than 3.
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    # Extract Title
    dataset["Title"]=dataset.Name.str.extract('([aA-Za-z]+)\.',expand=False )
pd.crosstab(train['Title'], train['Sex'])
  # replace title with most common title or as rare
for dataset in combine:
    dataset["Title"]=dataset["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"]=dataset["Title"].replace(['Mlle','Miss'])
    dataset["Title"]=dataset["Title"].replace(['Ms','Miss'])
    dataset["Title"]=dataset["Title"].replace(['Mme','Mra'])
    # Convert title into number 
    dataset["Title"]=dataset["Title"].map(titles)
   
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train['Cabin'].unique()
# Some thing like wing or floors in the ship
import re
deck={"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"U":0}
data=[train,test]
for dataset in data:
    dataset['Cabin']=dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck']=dataset['Deck'].map(deck)
    dataset['Deck']=dataset['Deck'].fillna(0)
    dataset['Deck']=dataset['Deck'].astype(int)
    
    
# we can now drop the cabin feature
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
# Now we can teckle the issue with the issue age feature 
print("Skewness is ",train['Age'].skew())
# hence data is skewed,now we have to impute the values 
data=[train,test]
for dataset in data:
    mean=dataset["Age"].mean()
    std=dataset["Age"].std()
    isnull=dataset["Age"].isnull().sum()
# random number between mean , std, null
    rand_age=np.random.randint(mean-std,mean+std,size=isnull)
# fill nan value in random value generated
    age_slicer=dataset["Age"].copy()
    age_slicer[np.isnan(age_slicer)] = rand_age
    dataset["Age"]=age_slicer
    dataset["Age"]=train["Age"].astype(int)
train["Age"].isnull().sum()
train["Embarked"].value_counts()
# 'S' has high frequency of values , so put "S" in high frequency position
common_value = 'S'
data = [train, test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train.info()
## Fare   -- Float
## Categorical variable
##       Name,Sex,Ticket,Embarked
# converting Fare from float to int64 using astype() Function
data=[train,test]

for dataset in data:
    dataset["Fare"]=dataset["Fare"].fillna(0)
    dataset["Fare"]=dataset["Fare"].astype(int)
data=[train,test]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # Extract Title
    dataset["Title"]=dataset.Name.str.extract('([aA-Za-z]+)\.',expand=False )
    # replace title with most common title or as rare
    dataset["Title"]=dataset["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"]=dataset["Title"].replace(['Mlle','Miss'])
    dataset["Title"]=dataset["Title"].replace(['Ms','Miss'])
    dataset["Title"]=dataset["Title"].replace(['Mme','Mra'])
    # Convert title into number 
    dataset["Title"]=dataset["Title"].map(titles)
    # fill na values
    dataset["Title"]=dataset["Title"].fillna(0)
# Droping name 
train=train.drop(["Name"],axis=1)
test=test.drop(["Name"],axis=1)
# Converting sex feature ibto numeric function
genders = {"male": 0, "female": 1}
data = [train, test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
train["Ticket"].describe()
# Since the Ticket attribute has 681 unique tickets, it will be a bit tricky to convert them into useful
# categories. So we will drop it from the dataset
train=train.drop(["Ticket"],axis=1)
test=test.drop(["Ticket"],axis=1)
train["Embarked"].unique()
ports={"S":0,"C":1,"Q":2}
data=[train,test]
for dataset in data:
    dataset["Embarked"]=dataset["Embarked"].map(ports)
data = [train, test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
# Lets check how it is distributed
train["Age"].value_counts()
data = [train, test]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train, test]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
data=[train,test]
for dataset in data:
    dataset["Fare per person"]=dataset["Fare"]/(dataset["Relatives"]+1)
    dataset["Fare per person"]=  dataset["Fare per person"].astype(int)
train.info()
test.info()
train=train.drop(["PassengerId"],axis=1)
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred1 = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train,Y_train) * 100, 2)
print(acc_gaussian)
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train,Y_train) * 100, 2)
print(acc_knn)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train) * 100, 2)
print(acc_svc)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, Y_train)
Y_pred5 = gbk.predict(X_test)
acc_gbk = round(gbk.score(X_train, Y_train) * 100, 2)
print(acc_gbk)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, Y_train)
Y_pred = decisiontree.predict(X_test)
acc_decisiontree = round(decisiontree.score(X_train,Y_train) * 100, 2)
print(acc_decisiontree)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, Y_train)
Y_pred = gbk.predict(X_test)
acc_gbk = round(gbk.score(X_train,Y_train) * 100, 2)
print(acc_gbk)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_logreg = round(logreg.score(X_train,Y_train) * 100, 2)
print(acc_logreg)
models = pd.DataFrame({
    'Model': ['Naive Bayes','KNN','Support Vector Machines', 'Gradient Boosting Classifier', 
              'Random Forest', 'Decision Tree','Logistic Regression', ],
    'Score': [acc_gaussian, acc_knn,acc_svc, acc_gbk,acc_logreg,acc_decisiontree,
              acc_logreg]})
models.sort_values(by='Score', ascending=False)
models
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred5
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)