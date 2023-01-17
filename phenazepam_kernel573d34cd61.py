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
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/train.csv")
validation = pd.read_csv("../input/test.csv")
# Check the train dataset.
print(train.columns.values)
train.describe(include='all')
print(train[['Pclass', 'Survived']].groupby(['Pclass']).mean())
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train)
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Fare")
group = pd.cut(train.Fare, [0,50,100,150,200,550])
piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Fare', aggfunc='count')
piv_fare.plot(kind='bar')
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
group = pd.cut(train.Age, [0,14,30,60,100])
piv_fare = train.pivot_table(index=group, columns='Survived', values = 'Age', aggfunc='count')
piv_fare.plot(kind='bar')
print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())
sns.catplot(x='Embarked', y='Survived',  kind='bar', data=train)
sns.catplot('Pclass', kind='count', col='Embarked', data=train)
print(train[['SibSp', 'Survived']].groupby(['SibSp']).mean())
sns.catplot(x='SibSp', y='Survived', data=train, kind='bar')
print(train[['Parch', 'Survived']].groupby(['Parch']).mean())
sns.catplot(x='Parch', y='Survived', data=train, kind='bar')
# Explanation for the next code block used to get the Titles
# Using this output to explain the string manipulation below
print(train.Name.head(1))
print()
# The above returns a single name for e.g. Braund, Mr. Owen Harris.
# Calling str returns a String object
print(train.Name.head(1).str)
print()
# Next we split the string into a List with a comma as the separator
print(train.Name.head(1).str.split(','))
print()
# Similary we remove the . and then strip the remaining string to get the title.
# We pick the second item of the
print(train.Name.head(1).str.split(',').str[1])
print()
# Get the titles
for dataset in [train, validation]:
    # Use split to get only the titles from the name
    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Check the initial list of titles.
    print(dataset['Title'].value_counts())
    print()
sns.catplot(x='Survived', y='Title', data=train, kind ='bar')
for df in [train, validation]:
    print(df.shape)
    print()
    print(df.isna().sum())
# Drop rows with nulls for Embarked
for df in [train, validation]:
    df.dropna(subset = ['Embarked'], inplace = True)
print(train[train['Fare'].isnull()])
print() 
# 1 row with null Fare in validation
print(validation[validation['Fare'].isnull()])
# We can deduce that Pclass should be related to Fares.
sns.catplot(x='Pclass', y='Fare', data=validation, kind='point')
validation['Fare'].fillna(validation[validation['Pclass'] == 3].Fare.median(), inplace = True)
print(train[['Age','Title']].groupby('Title').mean())
sns.catplot(x='Age', y='Title', data=train, kind ='bar')
train.drop(columns=['PassengerId'], inplace = True)
[df.drop(columns=['Ticket'], inplace = True) for df in [train, validation]]
[train, validation] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [train, validation]]
for df in [train, validation]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)
[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train, validation]]

train['Title'] = train['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
validation['Title'] = validation['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
[df.drop(columns=['Name'], inplace = True) for df in [train, validation]]
[train, validation] = [pd.get_dummies(data = df, columns = ['Title']) for df in [train, validation]]
print(train.columns.values)
print(validation.columns.values)
train.corr()
from sklearn.model_selection import train_test_split
# Use only the features with a coeefficient greater than 0.3
X = train[['Fare', 'Pclass_1', 'Pclass_3', 'Sex_female', 'Embarked_C', 'Embarked_S', 'HasCabin', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)
# We will also create a base model to check the goodness of our model.
# First we see the actual number of survivors
print(y.value_counts())
y_default = pd.Series([0] * train['Survived'].shape[0], name = 'Survived')
print(y_default.value_counts())

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y, y_default))
print()
print(accuracy_score(y, y_default))
# So if we assumed that everyone dies we would be correct 61% of the time.
# So this is the bare minimun level of accuracy our prediciton should aim to improve upon.
# First attempt with LinearSVC
from sklearn.svm import LinearSVC
# Looking into the documentation points us to set dual=False for cases with n_samples > n_features.
classifier = LinearSVC(dual=False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# Next up we will try KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# KNN isn't useful for us so we now move to a few popular ensemble estimators
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
print("AdaBoostClassifier")
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("BaggingClassifier")
classifier = BaggingClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("ExtraTreesClassifier")
classifier = ExtraTreesClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("GradientBoostingClassifier")
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("RandomForestClassifier")
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# Will also try XGB based on its popularity and relevance here.
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Print the confusion matrix
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
print(confusion_matrix(y_test, y_pred))
# Print the accuracy score
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
print(accuracy_score(y_test, y_pred))
X_validation = validation[['Fare', 'Pclass_1', 'Pclass_3', 'Sex_female', 'Embarked_C', 'Embarked_S', 'HasCabin', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]
# Call the predict from the created classifier
y_valid = classifier.predict(X_validation)
validation_pId = validation.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':validation_pId, 'Survived':y_valid})
print(my_submission['Survived'].value_counts())
my_submission.to_csv('submission.csv', index = False)