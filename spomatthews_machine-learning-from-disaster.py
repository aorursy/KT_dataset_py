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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Show where nulls are in columns for train set
print("-- Train Nulls --")
print("Survived Null Count: " + str(sum(train['Survived'].isna())))
print("PClass Null Count: " + str(sum(train['Pclass'].isna())))
print("Sex Null Count: " + str(sum(train['Sex'].isna())))
print("Age Null Count: " + str(sum(train['Age'].isna())))
print("SibSp Null Count: " + str(sum(train['SibSp'].isna())))
print("Parch Null Count: " + str(sum(train['Parch'].isna())))
print("Ticket Null Count: " + str(sum(train['Ticket'].isna())))
print("Fare Null Count: " + str(sum(train['Fare'].isna())))
print("Cabin Null Count: " + str(sum(train['Cabin'].isna())))
print("Embarked Null Count: " + str(sum(train['Embarked'].isna())))
males = train[train['Sex']=='male']
print("Males who don't show an Age: " + str(sum(males['Age'].isna())))
femalecount = 177-sum(males['Age'].isna())
print("Females who don't show an Age: " + str(femalecount))
#Show Number representation for Age split by Gender to replace Nans
print("Median Male Age: " + str(males[['Age']].median(axis=0)))
females = train[train['Sex']=='female']
print("Median Female Age: " + str(females[['Age']].median(axis=0)))
#Fill nulls with medians by sex
malenoage = train.loc[(train['Age'].isna()) & (train['Sex']=='male')]
rows = malenoage.index
train.loc[rows,'Age'] = 29
femalenoage= train.loc[(train['Age'].isna()) & (train['Sex']=='female')]
rows2=femalenoage.index
train.loc[rows2,'Age'] = 27
train
#Check nulls for Embarked
train[train['Embarked'].isna()]
#View if order of tickets relates to Embarked location
train[(train['Ticket']>'113500') & (train['Ticket']<'113600')]
#See how common the values are, as S is very present above, C is rare, and Q is non-existent
for x in train.Embarked.unique():
    train.loc[train['Embarked']==x].count()
#plot numerical correlations
train.corr()
#Extract title from name using regex
train['Title'] = train.Name.str.extract(', (\w{1,})\.')
train.Title
#The regex didn't catch one row, so I replaced it manually
train[train.Title.isna()]
train.loc[759,'Title'] = 'Countess'
#Examine Family Features Parch and SibSp
train[['Parch','SibSp']].describe()

import seaborn as sns

sns.distplot(train[['SibSp']])

#Chart shows that both SibSp and Parch have tons of 0, and many smaller families with a some rare large families
train[['Parch','Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)
train[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)
#Calculate total party size including self
train['FamilySize']=1+train['SibSp']+train['Parch']
train[['FamilySize','Survived']].groupby(['FamilySize']).mean().sort_values(by='Survived', ascending=False)
train.Pclass.hist(train.FamilySize)
train.FamilySize.hist(train.Pclass)
train[['FamilySize','Pclass','Survived']].groupby(['FamilySize','Pclass']).mean().sort_values(by='Survived', ascending=False)
train[(train['FamilySize']==3) | (train['FamilySize']==4)].Age.hist(figsize=(70,30), bins=7)
Survivors = train[train['Survived']==1]
FamilySurvivors = Survivors[(Survivors['FamilySize']==3) | (Survivors['FamilySize']==4)]
FamilySurvivors.Age.hist(figsize=(70,30), bins=7)
#According to the above table, Larger families may be drastically separated by also comparing the class they rode in.
#IE. First class large families survive, but poorer third class families do poorly. 
#It also appears that 3-4 size families survived at higher rates. But from the above charts, only the youngest and 
#oldest members benefited from the higher rate of survival
#Kids below 10 were likely the most protected group - and likely went with their mothers in most cases
AverageFamilies = train.loc[(train['FamilySize']>2) & (train['FamilySize']<5)]
AverageFamilies.Survived.mean() # 61% Survival vs. Population 38% Survival
AverageFamiliesWithoutDad = AverageFamilies.loc[((train['Age']>20) & (train['Sex']=='female') | (train['Age']<19))]
AverageFamiliesWithoutDad.Survived.mean() #77.4% Survival rate vs. 61% Survival with older males
AverageFamiliesWithoutDad.groupby('Pclass').Survived.mean() #1st and 2nd Class sported >90% Survival Rates
AverageFamilies.loc[(AverageFamilies['Sex']=='male') & (AverageFamilies['Age']>18)].Survived.mean() #20% Survival Rate for these men
train.loc[(train['Age']<1)].Survived.mean() #Babies survived at higher rates - 100%
train.loc[(train['Age']>1) & (train['Age']<16)].Survived.mean() #Kids between 1-16 survived at ~54% rates
train.loc[(train['Age']>40)].Survived.mean() #Older parties don't show any specifically higher levels of survival rate

#Create new columns for findings - Save the women and children!
train['NonPoorMothersAndChildren']= 0
train.loc[((train['Pclass']<3) & (train['FamilySize']>2) & (train['FamilySize']<5) & ((train['Age']>20) & (train['Sex']=='female') | (train['Age']<19))),'NonPoorMothersAndChildren']=1
train['IsBaby'] = 0
train.loc[train['Age']<1,'IsBaby']=1


#Let's explore the Titles feature
train.groupby('Title').Survived.mean() #The captain goes down with the ship!
train[['Title','Pclass','Survived']].groupby(['Title','Pclass']).Survived.mean()
# I didn't discover anything too useful here that would be a feature to add in and not cause overfitting
#Let's explore Fare costs
train['Fare'].hist() #This was the original graph that helped me understand why a fare of 50 and above would be a good starting place
train.loc[train['Fare']>50].Survived.mean() #68% survival rate Expensive Tickets
train.loc[train['Fare']<10].Survived.mean() #19% survival rate Cheap Tickets
train.loc[train['Fare']<50,'Fare'].hist()

#Let's make some categories!
train['CheapTickets'] = 0
train.loc[train['Fare']<10,'CheapTickets'] = 1
train['ExpensiveTickets'] = 0
train.loc[train['Fare']>50,'ExpensiveTickets'] = 1

#Let's explore Embarked location
train.groupby('Embarked').Survived.mean()
#Very interesting... People who embarked at C have significantly higher survival rates at 58%

train.groupby(['Embarked','Pclass']).Name.count()
#It appears that this is because most passengers who embarked at C were 1st class
#Let's explore the cabin column
train.loc[train['Cabin'].isna()].Survived.mean() #Those with no cabin have a 30% Survival Rate
train.loc[train['Cabin'].notnull()].Survived.mean() #People with a cabin show a 67% Survival Rate
train['HasCabin'] = 0
train.loc[train['Cabin'].notnull(),'HasCabin'] = 1
#Let's turn our categorical features into numerical features with dummies
train[['C','Q','S']] = pd.get_dummies(train['Embarked'])
train[['1Class','2Class','3Class']] = pd.get_dummies(train['Pclass'])
train[['Male','Female']] = pd.get_dummies(train['Sex'])
train[['Captain','Colonel','Countess','Don','Dr','Jonkheer','Lady','Major','Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir']] = pd.get_dummies(train['Title'])
#['Mr', 'Mrs', 'Miss', 'Master', 'Ms', 'Col', 'Rev', 'Dr', 'Dona'] are the titles in Test set
# So -'Captain', -'Countess', 'Don', 'Jonkheer', -'Lady', -'Major', -'Mme', -'Mlle', -'Sir' are not in the Test set
# We will have to group values accordingly
train['MilitaryTitle'] = 0
train.loc[((train['Colonel']==1) | (train['Captain']==1) | (train['Major']==1)),'MilitaryTitle'] = 1
train['FemaleTitle'] = 0
train.loc[((train['Mrs']==1) | (train['Ms']==1) | (train['Miss']==1) | (train['Lady']==1) | (train['Countess']==1)),'FemaleTitle'] = 1
train['RareTitle'] = 0
train.loc[(train['Jonkheer']==1) | (train['Don']==1),'RareTitle'] = 1
#Drop columns that are no longer useful
train = train.drop(['PassengerId','Embarked','Pclass','Sex','Cabin','Ticket','SibSp','Parch','Name','Title', 'Captain', 'Countess', 'Don', 'Jonkheer', 'Lady', 'Major', 'Mme', 'Mlle', 'Sir', 'Dr','Miss', 'Mr', 'Mrs', 'Ms', 'Colonel'], axis=1)
#Look at this beauty!
train.head()

#Now let's try some ML - For first round we will be using 
#RandomForestClassifier - It will be very useful considering that this data is very colinear

#First, we need to split the data into train/test
from sklearn.model_selection import train_test_split
y=train['Survived']
X = train.drop('Survived',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier(n_estimators=500)

rfclf.fit(X_train, y_train)
rfclf.score(X_test, y_test)

#Generally, after Random Forest do Logistic Regression
from sklearn.linear_model import LogisticRegression

logclf = LogisticRegression(solver='lbfgs', max_iter=1000)

logclf.fit(X_train, y_train)
logclf.score(X_test, y_test)
#Try Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gbclf = GradientBoostingClassifier(n_estimators=500)

gbclf.fit(X_train, y_train)
gbclf.score(X_test,y_test)
#Try AdaBoost
from sklearn.ensemble import AdaBoostClassifier

abclf = AdaBoostClassifier(n_estimators=500)

abclf.fit(X_train, y_train)
abclf.score(X_test,y_test)

#Try Support Vector Classifier
from sklearn.svm import SVC

svc = SVC(gamma='auto')

svc.fit(X_train, y_train)
svc.score(X_test, y_test)
rfclf_features = rfclf.fit(X_train,y_train).feature_importances_
logclf_features = logclf.fit(X_train,y_train).coef_
abclf_features = abclf.fit(X_train,y_train).feature_importances_
gbclf_features = gbclf.fit(X_train,y_train).feature_importances_
# Plot feature importance from Random Forest
cols = X_test.columns.values
plt.scatter(cols,rfclf_features)
plt.plot()

# Plot feature importance from Logistic Regression
plt.scatter(cols,logclf_features)
plt.plot()
# Plot feature importance from AdaBoost
plt.scatter(cols,abclf_features)
plt.plot()
# Plot feature importance from Gradient Boosting
plt.scatter(cols,gbclf_features)
plt.plot()
print("-- Test Nulls --")
print("PClass Null Count: " + str(sum(test['Pclass'].isna())))
print("Sex Null Count: " + str(sum(test['Sex'].isna())))
print("Age Null Count: " + str(sum(test['Age'].isna())))
print("SibSp Null Count: " + str(sum(test['SibSp'].isna())))
print("Parch Null Count: " + str(sum(test['Parch'].isna())))
print("Ticket Null Count: " + str(sum(test['Ticket'].isna())))
print("Fare Null Count: " + str(sum(test['Fare'].isna())))
print("Cabin Null Count: " + str(sum(test['Cabin'].isna())))
print("Embarked Null Count: " + str(sum(test['Embarked'].isna())))
males = test[test['Sex']=='male']
print("Males who don't show an Age: " + str(sum(males['Age'].isna())))
print("Median of Men's Age: " + str(males.Age.median()))
females = test[test['Sex']=='female']
print("Females who don't show an Age: " + str(sum(females['Age'].isna())))
print("Median of Women's Age: " + str(females.Age.median()))
#Since both are 27, replace both regardless of gender
test.loc[test['Age'].isna(),'Age'] = 27

#Check nulls for Embarked
test[test['Fare'].isna()]

#This is for a person from 3rd class who embarked at Southampton
#test.loc[(test['Pclass']==3)&(test['Embarked']=='S')].Fare.mean() # mean fare was 13.91

test.loc[test['Fare'].isna(),'Fare'] = 13.91
#Create dummies for HasCabin 
test['HasCabin'] = 0
test.loc[test['Cabin'].notnull(),'HasCabin'] = 1
#Extract title from name using regex
test['Title'] = test.Name.str.extract(', (\w{1,})\.')
test.loc[test['Title'].isna()]
#Calculate total party size including self
test['FamilySize']=1+test['SibSp']+test['Parch']
# Save the women and children!
test['NonPoorMothersAndChildren']= 0
test.loc[((test['Pclass']<3) & (test['FamilySize']>2) & (test['FamilySize']<5) & ((test['Age']>20) & (test['Sex']=='female') | (test['Age']<19))),'NonPoorMothersAndChildren']=1
test['IsBaby'] = 0
test.loc[test['Age']<1,'IsBaby']=1
#Let's make some more categories!
test['CheapTickets'] = 0
test.loc[test['Fare']<10,'CheapTickets'] = 1
test['ExpensiveTickets'] = 0
test.loc[test['Fare']>50,'ExpensiveTickets'] = 1
# Adjust for cabins
test['HasCabin'] = 0
test.loc[test['Cabin'].notnull(),'HasCabin'] = 1
#Let's turn our last categorical features into numerical features with dummies
test[['C','Q','S']] = pd.get_dummies(test['Embarked'])
test[['1Class','2Class','3Class']] = pd.get_dummies(test['Pclass'])
test[['Male','Female']] = pd.get_dummies(test['Sex'])
test[['Mr', 'Mrs', 'Miss', 'Master', 'Ms', 'Colonel', 'Rev', 'Dr', 'Dona']] = pd.get_dummies(test['Title'])

#['Mr', 'Mrs', 'Miss', 'Master', 'Ms', 'Col', 'Rev', 'Dr', 'Dona'] are the titles in Test set
# So -'Captain', -'Countess', 'Don', 'Jonkheer', -'Lady', -'Major', -'Mme', -'Mlle', -'Sir' are not in the Test set
# We will have to group values accordingly
test['MilitaryTitle'] = 0
test.loc[((test['Colonel']==1)),'MilitaryTitle'] = 1
test['FemaleTitle'] = 0
test.loc[((test['Mrs']==1) | (test['Ms']==1) | (test['Miss']==1) | (test['Dona']==1)),'FemaleTitle'] = 1
test['RareTitle'] = 0
          
#Drop columns that are no longer useful
test = test.drop(['PassengerId','Embarked','Pclass','Sex','Cabin','Ticket','SibSp','Parch','Name','Title','Miss', 'Mr', 'Mrs', 'Ms', 'Colonel', 'Dona', 'Dr'], axis=1)

test = test.sort_index(axis=1)
train = train.sort_index(axis=1)
train
'''#Produce Logistic Regression Results for the Test Data
logclf = LogisticRegression(solver='lbfgs', max_iter=1000)
y_train = train[['Survived']]
X_train = train.drop('Survived', axis=1)
logclf.fit(X_train, y_train)

test_results = logclf.predict(test)
test_results.mean()'''

#Produce Random Forest Results for the Test Data
rfclf = RandomForestClassifier(n_estimators=500)
y_train = train[['Survived']]
X_train = train.drop('Survived', axis=1)
rfclf.fit(X_train, y_train)

test_results = rfclf.predict(test)
test_results.mean()
submission = pd.DataFrame(test_results)
submission.to_csv('titanicsubmission1.csv')


