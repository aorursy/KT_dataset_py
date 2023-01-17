# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/gender_submission.csv')

train.head()
test.head()
sub.head()
# combine train and test dataset

df = train.append(test, ignore_index=True, sort=True)

print(df.info())

df.head(30)
# seperate cabin, SibSp, Parch, and Survived feature as another dataframe

_ = df[['Cabin', 'SibSp', 'Parch', 'Survived']]



# sum SibSp dan Parch as famsize then drop those two features

_['Famsize'] = _['SibSp'] + _['Parch']

_ = _.drop(['SibSp', "Parch"], axis=1)



# drop the missing value in cabin

_ = _[~_['Cabin'].isnull()]



# count how many unique cabin in it

print("There are total :", _['Cabin'].nunique(), "cabins in the data")



# make new columns to count if there any data that

_['Cabin Count'] = _['Cabin'].apply(lambda x: x.split(" "))

_['Cabin Count'] = _['Cabin Count'].apply(lambda x: len(x))



# print Cabin Count

print("Cabin Counts")

print(_['Cabin Count'].value_counts())
# show cabin different condition

print("Cabin with 4 data")

print(_[_['Cabin Count'] == 4])



print("Cabin with 3 data")

print(_[_['Cabin Count'] == 3])



print("Cabin with 2 data")

print(_[_['Cabin Count'] == 2])



print("Cabin with 1 data")

print(_[_['Cabin Count'] == 1][:20])
# make new feature as cabin code

_['Cabin Code'] = _['Cabin'].str[0]

_.head()
fig, ax = plt.subplots(1,2, figsize=(18,6))

sns.boxplot(x='Cabin Count', y='Famsize', data=_.sort_values("Cabin Count"), ax=ax[0])

sns.countplot(y='Cabin Code', hue='Survived', data=_[~_['Survived'].isnull()].sort_values('Cabin Code'), ax=ax[1])
# age categorization

def age_cat(x):

    if x<1:

        return 'Infant'

    elif x>=1 and x<12:

        return "Children"

    elif x>=12 and x<18:

        return "Teen"

    elif x>=18 and x<65:

        return "Adult"

    elif x>= 65:

        return "Senior Adult"



df['AgeCat'] = df['Age'].apply(lambda x: age_cat(x))



# Family Size

df['FamSize'] = df['SibSp'] + df['Parch'] + 1



# Honorific title

df['HonTit'] = df['Name'].apply(lambda x: re.findall('(\w+)\.',x)[0])



df.head()
# inspect Honorific Title that age are missing

fig, ax = plt.subplots(figsize=(18,6))

sns.countplot(x='HonTit', data=df[df['Age'].isnull()], ax=ax)

ax.set_title("Title Missing Age Count")
# plot age range and honorific title

fig, ax = plt.subplots(figsize=(18,6))

sns.boxplot(y='Age', x='HonTit', data=df, ax=ax)
# check the most Age Category for each Honorific Title

df.groupby(['HonTit','AgeCat'])['PassengerId'].count()
# fill the age category

def age_fill(x):

    if x == "Mr" or x == "Mrs" or x == "Miss" or x == "Dr" or x == "Ms":

        return "Adult"

    elif x == "Master":

        return "Children"



# make data slicing condition

adult = (df['HonTit'].str.contains('Mr|Mrs|Miss|Dr|Ms'))

child = (df['HonTit'] == 'Master')



# fill missing age category value

df.loc[adult,'AgeCat'] = df.loc[adult,'AgeCat'].fillna('Adult')

df.loc[child,'AgeCat'] = df.loc[child,'AgeCat'].fillna('Children')
# fill fare missing value with fare median

df['Fare'] = df['Fare'].fillna(df['Fare'].median())



# fill Embarked missing value with the most point of embarkment

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])



# check data condition

df.info()
# drop un-used features

_ = ['Age', 'Cabin', 'Name', 'SibSp', 'Parch', 'Ticket']

df = df.drop(_, axis=1)

df.info()
print(df['HonTit'].unique())
# change honorific title 

def hon_cat(x):

    if x in ['Lady', 'Sir', 'Countess', 'Jonkheer', 'Don', 'Dona']:

        return "Noble"

    elif x in ['Rev', 'Dr']:

        return "Professional"

    elif x in ['Major', 'Col', 'Capt']:

        return "Officer"

    else:

        return "Common"

    

df['HonTit'] = df['HonTit'].map(hon_cat)

df['HonTit'].value_counts()
# make some type changes to several features

df.loc[:,'Sex'] = df['Sex'].astype('category')

df.loc[:,'Embarked'] = df['Embarked'].astype('category')

df.loc[:,'Survived'] = df['Survived'].astype('category')

df.loc[:,'Pclass'] = df['Pclass'].astype('category')

df.loc[:,'FamSize'] = df['FamSize'].astype('category')

df.loc[:,'AgeCat'] = df['AgeCat'].astype('category')

df.loc[:,'HonTit'] = df['HonTit'].astype('category')

df.info()
# show numeric code representation from Sex and Embarked Feature Value

print(dict(zip(df['Sex'].cat.codes, df['Sex'])))

print(dict(zip(df['Embarked'].cat.codes, df['Embarked'])))

print(dict(zip(df['AgeCat'].cat.codes, df['AgeCat'])))

print(dict(zip(df['HonTit'].cat.codes, df['HonTit'])))



# make copy of dataframe to store the converted

df2 = df.copy()



# convert to numerical value

cat_col = df2.drop('Survived', axis=1).select_dtypes(['category']).columns

df2[cat_col] = df2[cat_col].apply(lambda x: x.cat.codes)

df2.head()
# calculate the correlation to data that survived value are exist

_ = df2[~df2['Survived'].isnull()]

_['Survived'] = _['Survived'].astype('int64')

_ = _.corr()



# plot the corrleation heatmap

fig, ax = plt.subplots(figsize=(18,5))

sns.heatmap(_, xticklabels=_.columns, yticklabels=_.columns, annot=True, fmt=".2f", ax=ax)
# plot distribution from features

fig, ax = plt.subplots(4, 2, figsize=(18,16))

sns.countplot(y='Pclass', data=df, ax=ax[0][0])

sns.violinplot(y='Fare',data=df, ax=ax[0][1])

sns.countplot(y='AgeCat', data=df, ax=ax[1][0])

sns.countplot(y='Sex', data=df, ax=ax[1][1])

sns.countplot(y='Embarked', data=df, ax=ax[2][0])

sns.countplot(y='FamSize', data=df, ax=ax[2][1])

sns.countplot(y='HonTit', data=df, ax=ax[3][0])

sns.countplot(y='Survived', data=df, ax=ax[3][1])
sns.catplot(x="Pclass", hue="Sex", col="Survived", data=df, kind="count");
fig, ax = plt.subplots(figsize=(18,4))

_ = df.groupby(['AgeCat'])['Survived'].value_counts(normalize=True).rename('percentage').mul(100).reset_index()

sns.barplot(x="AgeCat", y='percentage', hue="Survived", data=_, ax=ax)

ax.set_title("Surival Percentage per Age Category")

ax.grid(True, axis='y')
fig, ax = plt.subplots(1,2, figsize=(18,4))

sns.countplot(x="Pclass", hue="Survived", data=df[df['AgeCat'] == 'Children'], ax=ax[0])

sns.countplot(x="Pclass", hue="Survived", data=df[df['AgeCat'] == 'Teen'], ax=ax[1])



# give title

ax[0].set_title("Children Surival")

ax[1].set_title("Teen Surival")
fig, ax = plt.subplots(figsize=(18,4))

_ = df.groupby(['FamSize'])['Survived'].value_counts(normalize=True).rename('percentage').mul(100).reset_index()

sns.barplot(x="FamSize", y='percentage', hue="Survived", data=_, ax=ax)

ax.set_title("Surival Percentage per Passenger Family Size")

ax.grid(True, axis='y')
fig, ax = plt.subplots(1,2, figsize=(18,4))

_ = df.groupby(['HonTit'])['Survived'].value_counts(normalize=True).rename('percentage').mul(100).reset_index()

sns.barplot(x="HonTit", y='percentage', hue="Survived", data=_, ax=ax[0])

ax[0].set_title("Surival Percentage per Passenger Title")

ax[0].grid(True, axis='y')



_ = df.groupby(['HonTit'])['Pclass'].value_counts(normalize=True).rename('percentage').mul(100).reset_index()

sns.barplot(x="HonTit", y="percentage", hue='Pclass', data=_, ax=ax[1])

ax[1].set_title("Passenger Class Distribution per Title")

ax[1].grid(True, axis='y')
# import some algorithm

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# separete data from the dataframe that has missing value in survived feature as test dataset, and other as training dataset

X = df2[~df2['Survived'].isnull()].reset_index(drop=True)

y = X['Survived'].astype('int64')

test = df2[df2['Survived'].isnull()].reset_index(drop=True)



# drop survived column and passenger id columns

X = X.drop(['PassengerId','Survived', 'Embarked'], axis=1)

test = test.drop(['PassengerId', 'Survived', 'Embarked'], axis=1)



print("There are :", len(X), "data in training dataset")

print("There are :", len(test), "data in test dataset")
# split data into train and test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# prediction using LinReg Classifier

logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

y_pred_logreg = logistic_regression.predict(X_test)



# prediction using KNN

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)



# prediction using random forest

random_forest = RandomForestClassifier(n_estimators=100, random_state=1)

random_forest.fit(X_train, y_train)

y_pred_ranfor = random_forest.predict(X_test)
# Validation with mean absolute error

from sklearn.metrics import mean_absolute_error

print("MAE with Logistic Regression are: ", mean_absolute_error(y_test, y_pred_logreg))

print("MAE with KNN are: ", mean_absolute_error(y_test, y_pred_knn))

print("MAE with Random Forest are: ", mean_absolute_error(y_test, y_pred_ranfor))



# validation with sklearn accuracy score

from sklearn.metrics import accuracy_score

print("Accuracy score with Logistic Regression are: ", accuracy_score(y_test, y_pred_logreg))

print("Accuracy score with KNN are: ", accuracy_score(y_test, y_pred_knn))

print("Accuracy score with Random Forest are: ", accuracy_score(y_test, y_pred_ranfor))
# prediction using decision tree regressor

from sklearn.tree import DecisionTreeClassifier



# create function to create MAE from different max leaf to control underfitting / overfitting

def check_mae(max_leaf, X_train, X_test, y_train, y_test):

    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf, random_state=1)

    model.fit(X_train,y_train)

    y_pred_tree = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_tree)

    return mae



# loop over different max leaf nodes to get best MAE

for max_leaf in [5, 50, 500, 5000, 50000, 500000]:

    print("Max Leaf Nodes :", max_leaf, "\t\t Mean Absolute Error :", check_mae(max_leaf, X_train, X_test, y_train, y_test))
# set the max leaf nodes as 5000

tree_classifier = DecisionTreeClassifier(max_leaf_nodes=5000, random_state=1)

tree_classifier.fit(X_train, y_train)

y_pred_tree = tree_classifier.predict(X_test)

print("MAE with Decision Tree Regressor are: ", mean_absolute_error(y_test, y_pred_tree))
# try using logistic regression CV

from sklearn.linear_model import LogisticRegressionCV



# create function to create MAE from different CV to control underfitting / overfitting

def check_mae(num_CV, X_train, X_test, y_train, y_test):

    model = LogisticRegressionCV(cv=num_CV, random_state=1, multi_class='multinomial')

    model.fit(X_train,y_train)

    y_pred_tree = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_tree)

    return mae



# loop over different max leaf nodes to get best MAE

for num_cv in [5, 10, 15, 20]:

    print("Max Leaf Nodes :", num_cv, "\t\t Mean Absolute Error :", check_mae(num_cv, X_train, X_test, y_train, y_test))
# store prediction from logreg model

logistic_regression = LogisticRegressionCV(cv=5, random_state=1, multi_class='multinomial')

logistic_regression.fit(X,y)

y_pred = logistic_regression.predict(test)



# make new dataframe with predictions

submission = pd.DataFrame({'PassengerId': sub['PassengerId'], 'Survived': y_pred})



# save to csv

submission.to_csv('submission.csv', index=False)