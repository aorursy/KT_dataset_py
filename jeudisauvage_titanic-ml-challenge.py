#Import packages for analysis

import numpy as np

import pandas as pd

import sklearn as sk

#Import packages for visualisation

import seaborn as sns

import matplotlib.pyplot as plt
#Import the datasets 

dftest=pd.read_csv('/kaggle/input/titanic/test.csv')

dftrain=pd.read_csv('/kaggle/input/titanic/train.csv')

dftrain.head()
#Let's check how many null values we have in each column

print('Train columns with null values:\n', dftrain.isnull().sum())

print("-"*10)



print('Test/Validation columns with null values:\n', dftest.isnull().sum())

print("-"*10)



dftrain.describe(include = 'all')
#Since I intend to drop the cabin covariate, we need not worry about replacing the null values here

#However, the Age, Fare and Embarked missing values could be replaced

#I choose to replace the null Age & Fare values with the mean ages and fares respectively

#I also replace the missing embarked values with the mode as it is a categorical variable

datasets = [dftrain, dftest]

for dataset in datasets:

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace = True)

    

#Also, I wish to create a dummy variable corresponding to the passenger's title.

#We can use Name to create these

import re

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in datasets:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in datasets:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in datasets:

    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])

dftrain.head()

    
#Now, I drop the less releevant covariates and add my dummies

#I also wish to create dummy variables for 'Embarked', 'Sex' , 'Pclass' , 'Title' & 'Age_bin'

#I choose to drop names also.

#Below, I proceed to do this for both the training and testing sets

dftrain=dftrain.drop(columns=['Cabin', 'Ticket','Name','Age'])

dftest=dftest.drop(columns=['Cabin', 'Ticket','Name','Age'])

dftest1=pd.get_dummies(dftest, columns=['Embarked','Sex','Pclass','Title','Age_bin'],drop_first=True)

dftrain1=pd.get_dummies(dftrain, columns=['Embarked','Sex','Pclass','Title','Age_bin'],drop_first=True)

dftrain1=dftrain1.rename(columns={"Pclass_2": "Middle_Ticket_Class", "Pclass_3": "Lower_Ticket_Class"})

dftest1=dftest1.rename(columns={"Pclass_2": "Middle_Ticket_Class", "Pclass_3": "Lower_Ticket_Class"})

#Edit: After consideration I decided to engineer a new feature dummy 'Travelled_Alone'as well

#This is to combine the SibSp and Parch columns

datasets2 = [dftrain1, dftest1]

for dataset in datasets2:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'Travelled_Alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'Travelled_Alone'] = 1



    

dftrain1.head()
#Having created our new feature, I drop the Parch, relatives and SibSp columns for both datasets

dftrain2=dftrain1.drop(columns=['Parch', 'SibSp','relatives'])

dftest2=dftest1.drop(columns=['Parch','SibSp','relatives'])

#Here is our final dataset for analysis

dftrain2.head()

#I have created dummies for being male, location of embarkment & whether passenger travelled alone

#This means our reference category is females who departed from Cherbourg with an upper class ticket.

#Now, let's do a check for collinearity in our independent variables using a correlation heatmap.

xtrain = dftrain2[['Age_bin_Teenage','Age_bin_Adult','Age_bin_Elder','Title_Mr','Title_Miss','Title_Mrs','Title_Rare','Travelled_Alone','Fare','Embarked_Q','Embarked_S','Sex_male','Middle_Ticket_Class','Lower_Ticket_Class']]

xtest=dftest2[['Age_bin_Teenage','Age_bin_Adult','Age_bin_Elder','Title_Mr','Title_Miss','Title_Mrs','Title_Rare','Travelled_Alone','Fare','Embarked_Q','Embarked_S','Sex_male','Middle_Ticket_Class','Lower_Ticket_Class']]

corrmat = dftrain2.corr()

top_corr_features = corrmat.index

g=sns.heatmap(dftrain2[top_corr_features].corr(),annot=True,cmap="YlGnBu")

plt.show()

#From the above, there is no serious collinearity issues as all the correlations are quite low

#To implement a logistic regression model we need to convert xtrain to an array

#We also need to create an array for our dependent variable (Survived)

#From here, we employ a train/test split of 80/20 to test the accuracy of our training model

#This is because the test data is kept without values for 'Survived', and we do not want to overfit

from sklearn.model_selection import train_test_split

x1 = np.asarray(xtrain)

y1= np.asarray(dftrain2['Survived'])

X_train, X_test, Y_train, Y_test = train_test_split( x1, y1, test_size=0.2, random_state=1)

print ('Train set:', X_train.shape,  Y_train.shape)

print ('Test set:', X_test.shape,  Y_test.shape)
#Importing GBC package for classification

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.1,max_depth=4)

model.fit(X_train, Y_train)

yhat = model.predict(X_test)

#I import confusion matrix to measure the accuracy of the model

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(Y_test, yhat))

print(classification_report(Y_test,yhat))
#Now, predict the data, and prepare the result for submission

y_hat_test=model.predict(xtest)

dftest2.insert((dftest2.shape[1]),'Survived',y_hat_test)

dftest3=dftest2.drop(columns=['Age_bin_Teenage','Age_bin_Adult','Age_bin_Elder','Title_Mr','Title_Miss','Title_Mrs','Title_Rare','Travelled_Alone','Fare','Embarked_Q','Embarked_S','Sex_male','Middle_Ticket_Class','Lower_Ticket_Class'])

pid=dftest['PassengerId']

dftest3['PassengerId']= pid

dftest3.head()

dftest3.to_csv('results.csv',index=False)


