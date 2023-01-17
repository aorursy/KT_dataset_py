# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#data analysis libraries 

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
print(os.listdir("../input"))

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.
#import train and test CSV files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#take a look at the training data and it's shape
print(train.shape, test.shape)
train.describe(include="all")

print (train.columns)
train.head()

#Save the 'Id' column
train_ID = train['PassengerId']
test_ID = test['PassengerId']

train = train.drop('PassengerId',axis=1)
test = test.drop('PassengerId',axis=1)
corrmat = train.corr()
plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);

sns.countplot(x='Sex', hue='Survived', data=train, palette='RdBu')
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# What about the 'Embarked' feature ?
sns.countplot(x='Embarked', hue='Survived', data=train, palette='RdBu')
plt.xticks([0,1,2],['Southampton','Cherbourg ','Queenstown '])
# train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(train.Pclass ,train.Survived)
# train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Survived.values
all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.drop(['Survived'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
missingData = all_data.isnull().sum().sort_values(ascending=False)
percentageMissing = ((all_data.isnull().sum()/all_data.isnull().count())*100).sort_values(ascending=False)
totalMissing = pd.concat([missingData, percentageMissing], axis=1, keys=['Total','Percentage'])
totalMissing
all_data["hasCabin"] = (all_data["Cabin"].notnull().astype('int'))
sns.barplot(x="hasCabin", y="Survived", data=all_data)
plt.show()
all_data[['hasCabin', 'Survived']].groupby(['hasCabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# all_data = all_data.dropna('Cabin',axis=1)
# all_data = all_data.dropna('Embarked', axis=0)
all_data = all_data.drop('Cabin',axis=1)
all_data = all_data.drop('Ticket',axis=1)


# replacing the 2 missing values in the Embarked feature with S
# since majority of people embarked in Southampton (S)
all_data = all_data.fillna({"Embarked": "S"})


all_data.head()

all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(all_data['Title'], all_data['Sex'])
#get cell value: all_data.loc[0].at['Title']
#set cell value: all_data.at[0,'Title'] = 'Mr'

for i,row in all_data.iterrows():
    x = all_data.loc[i].at['Title']
    if x in ['Capt','Col','Don' ,'Dr' ,'Major','Rev' ,'Sir']:
        all_data.at[i,'Title']= 'Mr'
    if x in ['Mlle','Ms' ,'Dona' ,'Lady']:
        all_data.at[i,'Title']= 'Miss'
    if x in ['Countess','Jonkheer','Mme']:
        all_data.at[i,'Title'] = 'other'
        
pd.crosstab(all_data['Title'], all_data['Sex'])
allFemales = all_data[all_data['Sex']=='female'] # select all females
ThatOneFemale = allFemales[all_data['Title']=='Mr'] # select all females with title Mr
ThatOneFemale
# extracted the index of ThatOneFemale to be 796
all_data.at[796,'Title']='Mrs'

pd.crosstab(all_data['Title'], all_data['Sex'])
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "other": 5}
for i,row in all_data.iterrows():
    if all_data.loc[i].at['Title'] in title_mapping:
        all_data.at[i,'Title']= title_mapping[all_data.loc[i].at['Title']]
# all_data['Title']
all_data.head()
Mr_age = all_data[all_data['Title']==1].Age.mean()
Miss_age = all_data[all_data['Title']==2].Age.mean()
Mrs_age = all_data[all_data['Title']==3].Age.mean()
Master_age = all_data[all_data['Title']==4].Age.mean()
Other_age = all_data[all_data['Title']==5].Age.mean()
print(Mr_age, Miss_age, Mrs_age , Master_age, Other_age)

group_age_mapping = {1:Mr_age, 2: Miss_age, 3:Mrs_age, 4:Master_age, 5:Other_age}

for index,row in all_data.iterrows():
    if np.isnan(all_data.loc[index].at['Age']):
        all_data.at[index,'Age'] = group_age_mapping[all_data.loc[index].at['Title']]
        
all_data.drop('Name',axis=1,inplace=True)
sex_mapping = {"male": 0, "female": 1}
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

for i,row in all_data.iterrows():
    if all_data.loc[i].at['Sex'] in sex_mapping:
        all_data.at[i,'Sex']= sex_mapping[all_data.loc[i].at['Sex']]
    if all_data.loc[i].at['Embarked'] in embarked_mapping:
        all_data.at[i,'Embarked']= embarked_mapping[all_data.loc[i].at['Embarked']]
all_data.head()
mode = all_data['Fare'].mode() # extract the mode
all_data['Fare'].fillna(mode[0], inplace=True) # fill NaNs with the mode

all_data.drop('Survived',axis=1, inplace=True) # drop survived column

missingData = all_data.isnull().sum().sort_values(ascending=False)
percentageMissing = ((all_data.isnull().sum()/all_data.isnull().count())*100).sort_values(ascending=False)
totalMissing = pd.concat([missingData, percentageMissing], axis=1, keys=['Total','Percentage'])
totalMissing

all_data.head()
FareBand = pd.qcut(all_data['Fare'], 4)
FareBand.unique()
# get cell value: all_data.loc[0].at['Title']
# set cell value: all_data.at[0,'Title'] = 'Mr'

for i,row in all_data.iterrows():
    currFare=all_data.loc[i].at['Fare']
    if (currFare > -0.001 and currFare <=7.896):
        all_data.at[i,'Fare'] = 1
    if (currFare > 7.896 and currFare <=14.454):
        all_data.at[i,'Fare'] = 2
    if (currFare > 140454 and currFare <=31.275):
        all_data.at[i,'Fare'] = 3
    if (currFare > 31.275 and currFare <=512.329):
        all_data.at[i,'Fare'] = 4
        
all_data.head(10)
target = train['Survived']
trainData = all_data[0:ntrain]
testData = all_data[ntrain:]
target.shape, trainData.shape, testData.shape
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(trainData, target, test_size=0.2, random_state=0)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(accuracy_score(y_pred, y_test), 2)
print(acc_gaussian)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, y_test), 2)
print(acc_logreg)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_pred, y_test), 2)
print(acc_svc)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
y_pred = decisiontree.predict(X_test)
acc_decisiontree = round(accuracy_score(y_pred, y_test), 2)
print(acc_decisiontree)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) , 2)
print(acc_randomforest)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(y_pred, y_test) , 2)
print(acc_gbk)
#predictions for submission
predictions = gbk.predict(testData)
output = pd.DataFrame({ 'PassengerId' : test_ID, 'Survived': predictions })
output.to_csv('submission.csv', index=False)