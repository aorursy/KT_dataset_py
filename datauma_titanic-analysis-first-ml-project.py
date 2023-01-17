import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.info(), test.info()
train.isnull().sum()
test.isnull().sum()
sns.countplot('Survived', data=train)
sns.countplot(x='Survived', hue='Sex', data = train)
sns.countplot(x='Survived', hue='Pclass', data=train)
sns.catplot(x='Sex', hue='Pclass',col='Survived', data=train, kind='count')
train_test_data = [train, test]



for dataset in train_test_data:

    dataset['Title'] = dataset["Name"].str.extract( '([A-za-z]+)\.', expand=False )
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = { 'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':3, 'Rev':3, 'Mlle':3, 'Col':3, 'Major':3, 'Sir':3, 'Ms':3,

                 'Capt':3, 'Don':3, 'Jonkheer':3,'Countess':3,'Lady':3, 'Mme':3, 'Dona':3

    

}
for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train.head()
test.head()
#delete unnecessary columns from the dataset.



train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1,inplace=True)
train.head()
test.head()
train.info(), test.info()
from sklearn.preprocessing import LabelEncoder

labelEncoder_Sex = LabelEncoder()



for dataset in train_test_data:

    dataset['Sex'] = labelEncoder_Sex.fit_transform(dataset['Sex'])

train.head(), test.head()
#Filling missing values with median age for each title (Mr,Mrs, Miss, Others)



train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)

test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
train.head()
test.head()
for dataset in train_test_data:

    dataset.loc[dataset['Age']<= 16 , 'Age'] =0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <=26), 'Age' ] =1,

    dataset.loc[(dataset['Age'] >26) & (dataset['Age'] <= 36),'Age'] =2,

    dataset.loc[(dataset['Age'] >36 & (dataset['Age'] <=62),'Age')]=3,

    dataset.loc[(dataset['Age'] >62),'Age']=4
train.head()
test.head()
#Fill the missing values with the 'mode' of Embarkation



for dataset in train_test_data:

    mode_embarked = dataset['Embarked'].mode()[0]

    dataset['Embarked'].fillna(mode_embarked, inplace=True)
train.info(), test.info()
# Convert the Embarkation to categorical variables.



for dataset in train_test_data:

    dataset['Embarked'] = labelEncoder_Sex.fit_transform(dataset['Embarked'])
train.head()
test.head()
# Fill missing values of fare using median fare of each class.



test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)
test.info()
#Convert Fare to categorical values



for dataset in train_test_data:

    dataset.loc[dataset['Fare'] <=17, 'Fare'] =0,

    dataset.loc[(dataset['Fare'] >17) & (dataset['Fare'] <= 30), 'Fare'] =1,

    dataset.loc[(dataset['Fare'] >30) & (dataset['Fare'] <= 100), 'Fare'] =2,

    dataset.loc[dataset['Fare'] >100, 'Fare'] =3

    

    

    
train.head()
test.head()
#Fill out the missing values of cabin.



train['Cabin'].value_counts()
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
cabin_mapping = {'A': 0 , 'B':0.4 , 'C':0.8, 'D':1.2, 'E':1.6, 'F':2, 'G':2.4, 'T':2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
#fill out the missing values of cabin 



train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

train.info(),test.info()
train.head()
train['Cabin'].value_counts()
test.head()
test['Cabin'].value_counts()
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6 , 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4.0}



for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)



train.head()
test.head()
#Drop unnecessary columns from train and test data.

features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop('PassengerId', axis=1)
train_data = train.drop('Survived', axis =1)

target = train['Survived']
train_data.head()
target.head()
train.info(), test.info()


# Importing Classifier Modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
train.info()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors=13)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, scoring = scoring, n_jobs=1)

print(score)
#knn score



round(np.mean(score)*100,2)
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, scoring=scoring,n_jobs=1)

print(score)
#Decision Tree score

round(np.mean(score)*100,2)
clf = RandomForestClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, scoring=scoring, n_jobs=1, cv=k_fold)

print(score)
#Random forest classifier score



round(np.mean(score)*100,2)
clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, scoring=scoring, n_jobs=1)

print(score)
#Naive bayes score

round(np.mean(score)*100,2)
clf = SVC(gamma='auto')

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
#SVM score

round(np.mean(score)*100,2)
clf = SVC(gamma='auto')

clf.fit(train_data, target)



test_data = test.drop('PassengerId', axis=1).copy()

prediction = clf.predict(test_data)
submission = pd.DataFrame({

    'PassengerId':test['PassengerId'],

    'Survived':prediction

})



submission.to_csv('/kaggle/working/submission.csv', index=False)
submission = pd.read_csv('/kaggle/working/submission.csv')

submission.head()
print(os.listdir("/kaggle/working/"))