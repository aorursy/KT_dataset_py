#importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

from collections import Counter
#load data sets as train and test

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#detecting and removing outliers

def detect_and_remove_outliers(df):

    """this function takes in a dataframe df, checks columns specified below and 

    removes the row from the df if the number of outliers is more than 2"""

    outliers = []

    col = ["Age", "Parch", "SibSp", "Fare"]

    #checking interquartile range IQR for all columns

    for c in col:

        Q1 = df[c].quantile(0.25)

        Q3 = df[c].quantile(0.75)

        IQR = Q3 - Q1

        outliers.extend(df[(df[c] < Q1 - (1.5 * IQR)) | (df[c] > Q3 + (1.5 * IQR) )].index)

    #returning keys for count of occurrences in the list outlier key value pairs

    return  list(k for k,v in Counter(outliers).items() if v >2)

outliers = detect_and_remove_outliers(train)

outliers
#removing these outliers

#train = train.drop(outliers, axis=0).reset_index(drop=True)

train.info()
#concatenating the train and test data sets to ensure the features are comparable

Concatenated = pd.concat([train, test], axis=0, sort=True)

Concatenated.head()
#creating heatmap to visualize null values in different columns

sns.heatmap(Concatenated.isnull())

#the survived column may be ignored in this case since the values will be null for the test dataset.
#checking number of survivors by sex

sns.set_style('dark')

sns.countplot(data=Concatenated, x='Survived', hue='Sex')

#This plot indicates that if a person did not survive, they were more likely to be male.
#checking number of survivors by class

sns.countplot(data=Concatenated, x='Survived', hue='Pclass')

#This plot indicates that if a person did not survive, they were more likely to be in the 3rd class which had the cheapest tickets 

#available.
#checking if the port of embarkation had anything to do with a person surviving

sns.countplot(data=Concatenated, x='Survived', hue='Embarked')

#This indicates that most people boarded at port S. This may or may not have anything to do with their survival.
#checking to see if SibSp has a relationship with survival

sns.barplot(data=Concatenated, x='SibSp', y='Survived')

#this indicates that bigger families were less likely to survive
#checking to see if Parch has a relationship with survival

sns.barplot(data=Concatenated, x='Parch', y='Survived')

#this indicates that bigger families were less likely to survive
#creating a boxplot to determine average values of age by class

sns.boxplot(y='Age', x='Pclass', data=Concatenated)

Concatenated.groupby('Pclass', as_index=False)['Age'].mean()
#creating a function to replace null values in Age by Pclass

def rm_null_age(colnames):

    age = colnames[0]

    pclass = colnames[1]

    if pd.isnull(age):

        if pclass == 1:

            return 39

        elif pclass == 2:

            return 29

        else:

            return 25

    else:

        return age



Concatenated['Age'] = Concatenated[['Age', 'Pclass']].apply(rm_null_age, axis=1)
#creating heatmap to see where null values remain now

sns.heatmap(Concatenated.isnull())
#checking the relationship between age and survived

g = sns.FacetGrid(Concatenated, col='Survived')

g.map(sns.distplot, 'Age')
#checking Fare paid by Pclass

Concatenated.groupby('Pclass', as_index=False)['Fare'].mean()
#creating a function to handle null values in Fare

def rm_null_fare(colnames):

    fare = colnames[0]

    pclass = colnames[1]

    if pd.isnull(fare):

        if pclass == 1:

            return 86

        elif pclass == 2:

            return 21

        else:

            return 13

    else:

        return fare



Concatenated['Fare'] = Concatenated[['Fare', 'Pclass']].apply(rm_null_fare, axis=1)
#checking relationship with Fare

#sns.distplot(Concatenated['Fare'])

g = sns.FacetGrid(Concatenated, col='Survived', height=3, aspect=3)

g.map(sns.distplot, 'Fare')

#this indicates that those who paid a higher fare had a higher chance of survival.
#analyzing titles in name and adding them as a separate column in the dataframe

title = [item.split(', ')[1].split('.')[0] for item in Concatenated['Name']]

Concatenated['Title'] = pd.Series(title)

Concatenated.head()
#checking count for titles

Counter(Concatenated['Title'])
#marking titles with count less than 10 as rare

Concatenated['Title'] = Concatenated['Title'].replace(['Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir',

                                                       'Mlle', 'Col', 'the Countess', 'Jonkheer'], 'Rare')
#handling categorical variables

sex = pd.get_dummies(Concatenated['Sex'], drop_first=True)

embarked = pd.get_dummies(Concatenated['Embarked'], drop_first=True)

pclass = pd.get_dummies(Concatenated['Pclass'], drop_first=True)

title = pd.get_dummies(Concatenated['Title'], drop_first=True)

Concatenated = pd.concat([Concatenated, sex, embarked, pclass, title], axis=1)

Concatenated.head()
#removing all other columns that we do not need

Concatenated.drop(['Sex', 'Embarked', 'Pclass', 'Cabin', 'Ticket', 'Name', 'Title'], axis=1, inplace=True)

Concatenated.head()
#Splitting the test and train dataframes again

Test = Concatenated[Concatenated['Survived'].isnull()]

Train = Concatenated[-Concatenated['Survived'].isnull()]

Test.drop(['Survived'], axis=1, inplace=True)

Test.head()
Train['Survived'] = Train['Survived'].astype(int)

Train.head()
#dropping any remaining null rows

Train.dropna(axis=1, inplace=True)

sns.heatmap(Train.isnull())
#Splitting data set for cross validation

from sklearn.model_selection import train_test_split

X = Train.drop(['Survived'], axis=1)

y = Train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#Using Logistic Regression for prediction

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver='liblinear')

logmodel.fit(X_train, y_train)

predictionLR = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictionLR))
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, predictionLR)
#Using Random Forest for prediction

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(X_train, y_train)

prediction_RFC = rfc.predict(X_test)

print(classification_report(y_test, prediction_RFC))
roc_auc_score(y_test, prediction_RFC)
#Using Random forests since it gives a better prediction

predicted_RF = rfc.predict(Test)

predicted_RF = pd.DataFrame(predicted_RF)

predicted_RF.columns=['Survived']

Test_RFC = pd.concat([predicted_RF, Test], axis=1)

Test_RFC = Test_RFC[['PassengerId','Survived']]

Test_RFC['Survived'] = Test_RFC['Survived'].astype(int)

#Test_RFC.to_csv('Prediction_RFC2', index=False)
Test_RFC.head()
sns.heatmap(Test.isnull())
###Output through Logistic Regression

#predictionLR = logmodel.predict(Test)

#predictionLR = pd.DataFrame(predictionLR)

#predictionLR.columns = ['Survived']

#TestF = pd.concat([Test, predictionLR], axis=1)

#TestF['Survived'] = TestF['Survived'].astype(int)

#TestF = TestF.loc[:,['PassengerId', 'Survived']]

#TestF.head()

#TestF.to_csv('Prediction', index=False)
#Predicting model using Gender only

X_train_gender = pd.DataFrame(X_train['male'], columns = ['male'])

X_test_gender = pd.DataFrame(X_test['male'], columns = ['male'])

#logistic regression

logmodel_gender = LogisticRegression(solver='liblinear')

logmodel_gender.fit(X_train_gender, y_train)

prediction_gender = logmodel_gender.predict(X_test_gender)

print(classification_report(y_test, prediction_gender))

roc_auc_score(y_test, prediction_gender)



Test_gender = pd.DataFrame(Test[['male']], columns = ['male'])

prediction_gender = logmodel_gender.predict(Test_gender)

prediction_gender = pd.DataFrame(prediction_gender)

Test_gender = pd.concat([prediction_gender, Test['PassengerId']], axis=1)

Test_gender.columns = ['Survived', 'PassengerId']

Test_gender = Test_gender[['PassengerId','Survived']]

Test_gender.to_csv('Prediction_gender', index=False)