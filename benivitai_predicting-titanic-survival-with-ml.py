%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)



import pandas as pd

pd.options.display.max_columns = 100



from matplotlib import pyplot as plt

import numpy as np



import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv("/kaggle/input/titanic/test.csv")

data = pd.concat([train.drop('Survived',axis=1),test])
train.head()
plt.figure(figsize=(10,10))

sns.heatmap(data.isnull(),cmap="viridis",yticklabels=False,cbar=False)
train.info()
test.info()
train['PassengerId'].value_counts()
train['Name'].value_counts()
train['Cabin'].value_counts()
train['Ticket'].value_counts()
train = train.drop(['Cabin','Ticket','Name'],axis=1)

test = test.drop(['Cabin','Ticket','Name'],axis=1)
# rename columns to be more descriptive

train.rename(columns={"Pclass": "PClass", "Parch": "ParCh"},inplace=True)

test.rename(columns={"Pclass": "PClass", "Parch": "ParCh"},inplace=True)
# define default figsize helper function



def set_figsize():

    '''

    Sets default figsize to 12x8

    '''

    plt.figure(figsize=[12,8])
# define default legend helper function



def legend_survived():

    '''

    Plots legend with Not survived & Survived

    '''

    plt.legend(['Did not survive','Survived'],loc='best')
# create subsets of survived vs not_survived for hue in plots



survived = train[train['Survived'] == 1]

not_survived = train[train['Survived'] == 0]
set_figsize()

plt.title('Survival count by Sex')

sns.countplot('Sex',data=train,hue='Survived')

legend_survived()
set_figsize()

plt.title('Survival count by Passenger class')

sns.countplot('PClass',data=train,hue='Survived')

legend_survived()
set_figsize()

plt.title('Survival count by Port of Embarkation')

sns.countplot('Embarked',data=train,hue='Survived')

legend_survived()
set_figsize()

plt.title('Survival count by Number of Siblings/Spouse')

sns.countplot('SibSp',data=train,hue='Survived')

plt.xlabel('Number of Siblings/Spouse')

legend_survived()
set_figsize()

plt.title('Survival count by Number of Parents/Children')

sns.countplot('ParCh',data=train,hue='Survived')

plt.xlabel('Number of Parents/Children')

plt.legend(['Did not survive','Survived'],loc='upper right')
plt.figure(figsize=(20,8))



ax1 = sns.kdeplot(not_survived['Fare'],shade=True)

ax1.set_xlim((0,150))



ax2 = sns.kdeplot(survived['Fare'],shade=True)

ax2.set_xlim((0,150))



legend_survived()

plt.title('Survival density by Fare')

plt.xlabel('Fare')

plt.ylabel('Density')
plt.figure(figsize=(20,8))



ax1 = sns.kdeplot(not_survived['Age'],shade=True)



ax2 = sns.kdeplot(survived['Age'],shade=True)



legend_survived()

plt.title('Survival density by Age')

plt.xlabel('Age')

plt.ylabel('Density')
# Imputing Missing Age Values: choose median due to outliers, which affect mean

train['Age'].fillna(train['Age'].median(),inplace=True)



# Imputing Missing Embarked Values

train['Embarked'].fillna(train['Embarked'].value_counts().index[0], inplace=True)



#Creating a dictionary to convert Passenger Class from 1,2,3 to 1st,2nd,3rd.

d = {1:'1st',2:'2nd',3:'3rd'}



#Mapping the column based on the dictionary

train['PClass'] = train['PClass'].map(d)



# Getting Dummies of Categorical Variables

cat_vars = train[['PClass','Sex','Embarked']]

dummies = pd.get_dummies(cat_vars,drop_first=True)



# Drop original cat_vars

train = train.drop(['PClass','Sex','Embarked'],axis=1)

# Concatenate dummies and train

train = pd.concat([train,dummies],axis=1)



# Check the clean version of the train data.

train.head()
# split features and label

X = train.drop(['Survived'],1)

y = train['Survived']



# Use train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# choose GBC

# make predictions

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(learning_rate=0.1,max_depth=3)

GBC.fit(X_train, y_train)

pred_GBC = GBC.predict(X_test)
# evaluate performance

from sklearn.metrics import confusion_matrix, classification_report

print("GBC results:\n")

print(confusion_matrix(y_test, pred_GBC))

print(classification_report(y_test,pred_GBC))
# Try RFC

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC.fit(X_train, y_train)

pred_RFC = RFC.predict(X_test)
# evaluate performance

print("RFC results:\n")

print(confusion_matrix(y_test, pred_RFC))

print(classification_report(y_test,pred_RFC))
# Try logistic regression

from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()

logit.fit(X_train, y_train)

pred_logit = logit.predict(X_test)
# evaluate performance

print("Logistic Regression results:\n")

print(confusion_matrix(y_test, pred_logit))

print(classification_report(y_test,pred_logit))
# SVC

from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

pred_svc = svc.predict(X_test)
# evaluate performance

print("SVC results:\n")

print(confusion_matrix(y_test, pred_svc))

print(classification_report(y_test,pred_svc))
# Imputing Missing Age Values: choose median due to outliers, which affect mean

test['Age'].fillna(test['Age'].median(),inplace=True)



# Imputing Missing Embarked Values

test['Fare'].fillna(test['Fare'].median(), inplace=True)



# Impute Embarked

test['Embarked'].fillna(test['Embarked'].value_counts().index[0], inplace=True)

#Creating a dictionary to convert Passenger Class from 1,2,3 to 1st,2nd,3rd.

d = {1:'1st',2:'2nd',3:'3rd'}



#Mapping the column based on the dictionary

test['PClass'] = test['PClass'].map(d)



# Getting Dummies of Categorical Variables

cat_vars = test[['PClass','Sex','Embarked']]

dummies = pd.get_dummies(cat_vars,drop_first=True)



# Drop original cat_vars

test = test.drop(cat_vars,axis=1)

# Concatenate dummies and train

test = pd.concat([test,dummies],axis=1)



idx = test[['PassengerId']]

preds = model.predict(test)

results = idx.assign(Survived=preds)

results.to_csv('GBC_submission.csv',index=False)