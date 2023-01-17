# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

both = [train_df, test_df]
train_df.info()
train_df.head()
print(str(len(train_df.index)))
sns.countplot(x="Survived", data=train_df)
sns.countplot(x="Survived", hue="Sex", data=train_df)

# As shown in plot Females are more than the Males who survived!
sns.countplot(x="Survived", hue="Pclass", data=train_df)

# first class pessanger has the better surviving rate
train_df.hist(column="Age")
train_df.info()
sns.countplot(x='SibSp', data=train_df)
sns.countplot(x='Parch', data=train_df)
train_df.isnull().sum()
test_df.isnull().sum()
sns.heatmap(data=test_df.isnull() )


sns.heatmap(data=train_df.isnull(), cmap='viridis', yticklabels=False)
sns.boxplot(x='Pclass', y='Age', data=train_df)

def processing_age(cols):

    Age= cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        if Pclass ==2:

            return 29

        else:

            return 24

    else:

        return Age

train_df['Age'] = train_df[['Age', 'Pclass']].apply(processing_age, axis=1)
# from sklearn.preprocessing import Imputera
# imputer = Imputer(missing_values="NaN", strategy ='mean', axis=0)
train_df.head(6)
train_df.drop('Cabin', axis=1, inplace=True)
# meanage = train_df["Age"].mean()
# /meanage
# medianage = train_df['Age'].median()

# medianage
# train_df['Age'].fillna(value=medianage, axis=0, inplace=True)

# train_df['Age'].fillna(method = "bfill", axis=0, inplace=True)
# imputer = imputer.fit(train_df.iloc[:, 5:11])


# train_df.dropna(inplace=True)

sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
# meanagetest = test_df["Age"].mean()
# medianagetest = test_df['Age'].median()

# medianagetest
test_df.isnull().sum()
test_df['Age'] = test_df[['Age', 'Pclass']].apply(processing_age, axis=1)
# test_df['Age'].fillna(value=medianagetest, axis=0, inplace=True)

# test_df['Age'].fillna(method = "bfill", axis=0, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

# test_df.dropna(inplace=True)

final = test_df

sns.heatmap(test_df.isnull(), cbar=False)
train_df.isnull().sum()

test_df.isnull().sum()
pclass = pd.get_dummies(train_df['Pclass'], drop_first=True)
sex = pd.get_dummies(train_df['Sex'], drop_first=True)
embark =  pd.get_dummies(train_df['Embarked'], drop_first=True)
embark.head()
train_df=pd.concat([train_df, embark, sex, pclass], axis=1)
train_df.info()
train_df.head()
pclass = pd.get_dummies(test_df['Pclass'], drop_first=True)

sex = pd.get_dummies(test_df['Sex'], drop_first=True)

embark =  pd.get_dummies(test_df['Embarked'], drop_first=True)

test_df=pd.concat([test_df, embark, sex, pclass], axis=1)

test_df.head()
train_df.drop(['Pclass', 'Embarked', 'PassengerId','Ticket', 'Name', 'Sex', 'Fare'], axis=1, inplace=True)
train_df.to_csv("titanic_cleaned_data.csv", index=False)
# train_df.drop('Ticket', axis=1, inplace=True)

train_df.head()
test_df.drop(['Pclass', 'Embarked', 'PassengerId', 'Ticket','Name', 'Sex', 'Fare'], axis=1, inplace=True)

test_df.to_csv("cleaned_test_df.csv")

# test_df.drop('Ticket', axis=1, inplace=True)

# test_df.head()
test_df.head()
X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

randomfpredictor = clf.predict(X_test)

sc0=accuracy_score(y_true=y_test, y_pred=randomfpredictor)

print("Accuracy Score: ", accuracy_score(y_true=y_test, y_pred=randomfpredictor))
rfprediction = clf.predict(test_df)

print(confusion_matrix(y_test, randomfpredictor))

print(classification_report(y_test, randomfpredictor))
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(X_train, y_train)

survived = reg.predict(X_test)

sc1=accuracy_score(y_true=y_test, y_pred=survived)

print("Accuracy Score: ", accuracy_score(y_true=y_test, y_pred=survived))
prediction = reg.predict(test_df)

print(confusion_matrix(y_test, survived))

print(classification_report(y_test, survived))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

survivedknn = knn.predict(X_test)

sc2=accuracy_score(y_true=y_test, y_pred=survivedknn)

print("Accuracy Score: ", accuracy_score(y_true=y_test, y_pred=survivedknn))
predictionknn = knn.predict(test_df)

print(confusion_matrix(y_test, survivedknn))

print(classification_report(y_test, survivedknn))
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

survivedsvc = knn.predict(X_test)

sc3=accuracy_score(y_true=y_test, y_pred=survivedsvc)

print("Accuracy Score: ", accuracy_score(y_true=y_test, y_pred=survivedsvc))
predsvc = knn.predict(test_df)

print(confusion_matrix(y_test, survivedsvc))

print(classification_report(y_test, survivedsvc))
from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.1, 1, 10, 100], 'gamma':[ 1,0.1, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=2)

grid.fit(X_train, y_train)
print(grid.best_params_)

# print(grid.best_estimator_)

sc4=grid.best_score_

print(grid.best_score_)

gridpred = grid.predict(X_test)

print(confusion_matrix(y_test, gridpred))

print(classification_report(y_test, gridpred))
dict1 = {'Logistic Regression':sc1, 'Random Forest':sc0, 'Support vector classifier':sc2, 'KNearestNeighbours':sc3}
models=pd.DataFrame({"Models":['Logistic Regression', 'Random Forest', 'Support vector classifier', 'KNearest Neighbours'], 

             'Score':[sc1,sc0, sc4, sc3]

             })



models.sort_values(by='Score', ascending=False)

final.drop(['Pclass', 'Name', 'Sex','Age', 'SibSp', 'Ticket', 'Fare', 'Embarked', 'Parch'], axis=1, inplace=True)

randomFinal = final

svcfinal  = final

randomFinal["Survived"] = rfprediction

randomFinal.to_csv('randomForestResult.csv', index=False)
final['Survived'] = prediction

final.to_csv('Subbmission.csv', index=False)
svcfinal['Survived'] = prediction

final.to_csv('svcsurvived.csv', index=False)
# final.head()