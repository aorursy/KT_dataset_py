# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import precision_recall_fscore_support



# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# Get the info of the data frame

titanic_df.info()

# There are 12 features and 891 rows

# Null Columns - Age, Cabin, Embarked



# Add one more column called Salutation

conditions = [

    (titanic_df['Name'].str.contains('Master')),

    (titanic_df['Name'].str.contains('Mrs.')),

    (titanic_df['Name'].str.contains('Mr.')),

    (titanic_df['Name'].str.contains('Miss'))]

choices = ['Master', 'Mrs', 'Mr', 'Miss']

titanic_df["Salutation"] = np.select(conditions, choices, default='')



means = titanic_df.groupby('Salutation').mean()

print(means["Age"]["Master"])

titanic_df.loc[titanic_df.Salutation == 'Master','Age'] = titanic_df['Age'].fillna(means["Age"]["Master"]) 

titanic_df.loc[titanic_df.Salutation == 'Mrs','Age'] = titanic_df['Age'].fillna(means["Age"]["Mrs"]) 

titanic_df.loc[titanic_df.Salutation == 'Mr','Age'] = titanic_df['Age'].fillna(means["Age"]["Mr"]) 

titanic_df.loc[titanic_df.Salutation == 'Miss','Age'] = titanic_df['Age'].fillna(means["Age"]["Miss"]) 



# Change Sex to identify kids/teens info

titanic_df.loc[titanic_df.Salutation == 'Master', 'Sex'] = 'master'

titanic_df.loc[titanic_df.Salutation == 'Miss', 'Sex'] = 'miss'



# Drop Name and Ticket columns

titanic_df = titanic_df.drop(['Name','Ticket','Salutation'], axis=1)





# Fill NA values of Age Cabin and Embarked

#1] Calc kids average age



titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

titanic_df['Embarked'].fillna('B' , inplace=True)

titanic_df['Cabin'].fillna('Z' , inplace=True)



# Take only first letter of Cabin 

titanic_df['Cabin'] = titanic_df['Cabin'].astype(str).str[0]



# Convert Cabin, Embarked and  Sex into Numeric

titanic_df['Sex'].replace(['female','male','master','miss'], [0,1,2,3],inplace=True)

titanic_df['Cabin'].replace(['Z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], [0,1,2,3,4,5,6,7,8],inplace=True)

titanic_df['Embarked'].replace(['B','S','C','Q'], [0,1,2,3],inplace=True)



# Change Age to float

titanic_df['Age']  =  titanic_df['Age'].astype(float)

#Check the clenaed dataframe

titanic_df.info()

titanic_df.head()

# define training and testing sets

X = titanic_df.drop(["Survived", "PassengerId"],axis=1)

y = titanic_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)



#X = (X - X.mean()) / (X.max() - X.min())

# Build a forest and compute the feature importances

forest = ExtraTreesClassifier()

forest.fit(X, y)



importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("feature %d. %s : (%f)" % (indices[f],X.columns.values[indices[f]], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="g", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()



# Note that features lile Sex, Fare and Age are very important followed by Pclass, cabin, parch, sibsp and Embarked
sns.swarmplot(x='Sex', y='PassengerId',data=titanic_df, hue='Survived')



facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, titanic_df['Fare'].max()))

facet.add_legend()
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, titanic_df['Age'].max()))

facet.add_legend()
sns.swarmplot(x='Pclass', y='PassengerId',data=titanic_df, hue='Survived')
sns.countplot(x='Cabin',data=titanic_df, hue='Survived')
sns.countplot(x='SibSp',data=titanic_df, hue='Survived')

sns.countplot(x='Parch',data=titanic_df, hue='Survived')
sns.countplot(x='Embarked',data=titanic_df, hue='Survived')
#Drop Embarked and PArch columns it is not contributing much

X = X = titanic_df.drop(["Survived", "PassengerId","Parch"],axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)



# Random Forests

random_forest = RandomForestClassifier(max_depth=5, random_state=0)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

testdata = y_test

predictions = Y_pred



# save confusion matrix and slice into four pieces

confusion = metrics.confusion_matrix(testdata, predictions)

print('Confusion Metrics')

print(confusion)

print('Model Accuracy =', metrics.accuracy_score(testdata, predictions))



#[row, column]

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

precision = TP / float(TP + FP)

recall = TP / float(FN + TP)

print('Precision=',precision)

print('Recall=',recall)

# Finally train the model on entire training data

random_forest.fit(X, y)

# Get the info of the data frame

test_ToPred = test_df

test_ToPred.info()

# There are 12 features and 891 rows

# Null Columns - Age, Cabin, Embarked



# Add one more column called Salutation

conditions = [

    (test_ToPred['Name'].str.contains('Master')),

    (test_ToPred['Name'].str.contains('Mrs.')),

    (test_ToPred['Name'].str.contains('Mr.')),

    (test_ToPred['Name'].str.contains('Miss'))]

choices = ['Master', 'Mrs', 'Mr', 'Miss']

test_ToPred["Salutation"] = np.select(conditions, choices, default='')



means = test_ToPred.groupby('Salutation').mean()



test_ToPred.loc[test_ToPred.Salutation == 'Master','Age'] = test_ToPred['Age'].fillna(means["Age"]["Master"]) 

test_ToPred.loc[test_ToPred.Salutation == 'Mrs','Age'] = test_ToPred['Age'].fillna(means["Age"]["Mrs"]) 

test_ToPred.loc[test_ToPred.Salutation == 'Mr','Age'] = test_ToPred['Age'].fillna(means["Age"]["Mr"]) 

test_ToPred.loc[test_ToPred.Salutation == 'Miss','Age'] = test_ToPred['Age'].fillna(means["Age"]["Miss"]) 



# Change Sex to identify kids/teens info

test_ToPred.loc[test_ToPred.Salutation == 'Master', 'Sex'] = 'master'

test_ToPred.loc[test_ToPred.Salutation == 'Miss', 'Sex'] = 'miss'



# Drop Name and Ticket columns

test_ToPred = test_ToPred.drop(['PassengerId','Parch','Name','Ticket','Salutation'], axis=1)





# Fill NA values of Age Cabin and Embarked

#1] Calc kids average age



test_ToPred['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

test_ToPred['Fare'].fillna(titanic_df['Fare'].mean(), inplace=True)

test_ToPred['Embarked'].fillna('B' , inplace=True)

test_ToPred['Cabin'].fillna('Z' , inplace=True)



# Take only first letter of Cabin 

test_ToPred['Cabin'] = titanic_df['Cabin'].astype(str).str[0]



# Convert Cabin, Embarked and  Sex into Numeric

test_ToPred['Sex'].replace(['female','male','master','miss'], [0,1,2,3],inplace=True)

test_ToPred['Cabin'].replace(['Z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], [0,1,2,3,4,5,6,7,8],inplace=True)

test_ToPred['Embarked'].replace(['B','S','C','Q'], [0,1,2,3],inplace=True)



# Change Age to float

test_ToPred['Age']  =  titanic_df['Age'].astype(float)

#Check the clenaed dataframe

test_ToPred.info()

test_ToPred.head()

Y_pred = random_forest.predict(test_ToPred)

test_op = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

test_op.to_csv('titanicPrediction.csv', index=False)

test_op.head()