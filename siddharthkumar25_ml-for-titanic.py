#IMPORTING THE DEPENDENCIES



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.shape
df_train.head()
df_train.info()
df_test.head()
df_test.shape
df_test.info()
import missingno as mn
print("\t\tMISSING VALUES IN TRAINING DATA")

for i in df_train.columns:

    print("Missing Values In {} : {}".format(i,df_train[i].isnull().sum()))
mn.matrix(df_train)
mn.bar(df_train)
print("\t\tMISSING VALUES IN TESTING DATA")

for i in df_test.columns:

    print("Missing Values In {} : {}".format(i,df_test[i].isnull().sum()))
mn.matrix(df_test)
mn.bar(df_test)
print("\tTRAINING DATA")

print("The number of missing values in Cabin: {}".format(df_train['Cabin'].isnull().sum()))

print("The percentage of missing values in Cabin: {} %".format(df_train['Cabin'].isnull().sum()*100/891))

print("")

print("\tTESTING DATA")

print("The number of missing values in Cabin: {}".format(df_test['Cabin'].isnull().sum()))

print("The percentage of missing values in Cabin: {} %".format(df_test['Cabin'].isnull().sum()*100/418))

#Filling the unknown cabin with 'U'

df_train['Cabin'].fillna(value='U',inplace=True)
#Using only the letter of the Cabin without the number

df_train['CabinType'] = df_train['Cabin'].apply(lambda i: i[:1])
#Similar for testing Data

df_test['Cabin'].fillna(value='U',inplace=True)

#Using only the letter of the Cabin without the number

df_test['CabinType'] = df_test['Cabin'].apply(lambda i: i[:1])
print("\tTRAINING DATA")

print("The number of missing values in Embarked: {}".format(df_train['Embarked'].isnull().sum()))

print("The percentage of missing values in Embarked: {} %".format(df_train['Embarked'].isnull().sum()*100/891))

print("")

print("\tTESTING DATA")

print("The number of missing values in Embarked: {}".format(df_test['Embarked'].isnull().sum()))

print("The percentage of missing values in Embarked: {} %".format(df_test['Embarked'].isnull().sum()*100/418))

embarked_common = df_train['Embarked'].value_counts().index[0]

df_train['Embarked'].fillna(value=embarked_common,inplace=True)
print("\tTRAINING DATA\t")

print("The number of missing values in Fare: {}".format(df_train['Fare'].isnull().sum()))

print("The percentage of missing values in Fare: {}".format(df_train['Fare'].isnull().sum()*100/889))



print("")

print("\tTESTING DATA\t")

print("The number of missing values in Fare: {}".format(df_test['Fare'].isnull().sum()))

print("The percentage of missing values in Fare: {}".format(df_test['Fare'].isnull().sum()*100/418))

df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
df_train['Title'] = df_train['Name'].apply(lambda i: i.split(',')[1].split('.')[0].strip())

df_train.head()
standardized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}
df_train['Title'] = df_train['Title'].map(standardized_titles)
df_train.head()
#Grouping Sex,Pclass and Title Together

df_grouped = df_train.groupby(['Sex','Pclass', 'Title'])
df_grouped['Age'].median()
df_train['Age'] = df_grouped['Age'].apply(lambda i: i.fillna(i.median()))          
#Same procedure for testing data's Age column



df_test['Title'] = df_test['Name'].apply(lambda i: i.split(',')[1].split('.')[0].strip())

df_test['Title'] = df_test['Title'].map(standardized_titles)

df_grouped_test = df_test.groupby(['Sex','Pclass', 'Title'])

df_test['Age'] = df_grouped_test['Age'].apply(lambda i: i.fillna(i.median()))          

df_test['Age'].isnull().sum()
print("\t\tMISSING VALUES IN TRAINING DATA")

for i in df_train.columns:

    print("Missing Values In {} : {}".format(i,df_train[i].isnull().sum()))

    

print("")

print("\t\tMISSING VALUES IN TESTING DATA")

for i in df_test.columns:

    print("Missing Values In {} : {}".format(i,df_test[i].isnull().sum()))
#Storing the passengerId for future submissions.

passengerId = df_test['PassengerId']
df_titanic = pd.DataFrame()

df_titanic = df_train.append(df_test)
train_index = len(df_train)

test_index = len(df_titanic) - len(df_test)
df_titanic.head()
print("\t\tMISSING VALUES IN COMBINED DATA")

for i in df_titanic.columns:

    print("Missing Values In {} : {}".format(i,df_titanic[i].isnull().sum()))
df_titanic['FamilySize'] = df_titanic['Parch'] + df_titanic['SibSp'] + 1
df_titanic['Sex'] = df_titanic['Sex'].map({"male": 0, "female":1})
PClass_dummy = pd.get_dummies(df_titanic['Pclass'], prefix="Pclass")

Title_dummy = pd.get_dummies(df_titanic['Title'], prefix="Title")

CabinType_dummy = pd.get_dummies(df_titanic['CabinType'], prefix="CabinType")

Embarked_dummy = pd.get_dummies(df_titanic['Embarked'], prefix="Embarked")
df_titanic_final = pd.DataFrame()

df_titanic_final = pd.concat([df_titanic, PClass_dummy, Title_dummy, Embarked_dummy,CabinType_dummy], axis=1)
df_titanic_final.head()
df_train_final = df_titanic_final[ :train_index]

df_test_final = df_titanic_final[test_index: ]
#If not converted to 'int', can result in 0 score after submission as it doesn't get converted to boolean.

df_train_final.Survived = df_train_final.Survived.astype(int)
df_train_final.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

df_train_final.columns
df_test_final.columns


X = df_train_final.drop(['Cabin', 'CabinType', 'Embarked','Name',

       'PassengerId', 'Pclass','Survived', 'Ticket', 'Title'], axis=1).values 

Y = df_train_final['Survived'].values
df_test_final.columns
X_test = df_test_final.drop(['Cabin', 'CabinType', 'Embarked','Name',

       'PassengerId', 'Pclass','Survived', 'Ticket', 'Title'], axis=1).values
parameters_dict = dict(     

    max_depth = [n for n in range(10, 21)],     

    min_samples_split = [n for n in range(5, 11)], 

    min_samples_leaf = [n for n in range(2, 5)],     

    n_estimators = [n for n in range(10, 70, 10)],

)
rfc = RandomForestClassifier()
forest_gridcv = GridSearchCV(estimator=rfc, param_grid=parameters_dict, cv=5) 

forest_gridcv.fit(X, Y)
print("Best score: {}".format(forest_gridcv.best_score_))

print("Optimal params: {}".format(forest_gridcv.best_estimator_))
rfc_predictions = forest_gridcv.predict(X_test)
rfc_predictions
kaggle_final = pd.DataFrame({'PassengerId': passengerId, 'Survived': rfc_predictions})
#save to csv

kaggle_final.to_csv('mysubmission3.csv', index=False)