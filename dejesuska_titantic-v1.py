# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the training dataset

csv_path='../input/titanic/train.csv'

df_train = pd.read_csv(csv_path)

df_train.head()
# Check the data types to make sure nothing is improperly typed

df_train.dtypes
# Get a statistical overview of the data

df_train.describe(include='all')
# Check the number of missing data points for each column

df_train.isnull().sum()
# Drop the Cabin column

df_train.drop('Cabin',axis=1,inplace=True)

df_train.head()
# Calculate the mean age of the passengers

mean_age = df_train['Age'].mean()

mean_age
# Fill in the missing age values with the mean

df_train['Age'].replace(np.nan, mean_age, inplace=True)

df_train['Age'].isnull().sum()
# Replace the missing values with "U"

df_train['Embarked'].replace(np.nan, 'U', inplace=True)

df_train['Embarked'].isnull().sum()
# Get a summary of the survival number by passenger class

df_survived_grp=df_train.groupby(['Pclass','Survived']).size().unstack()

df_survived_grp
# Create a bar chart of the passenger class survival numbers

df_survived_grp.plot(kind='bar', stacked=True)
# Create a bar chart of survival by sex

df_survived_grp=df_train.groupby(['Sex','Survived']).size().unstack()

df_survived_grp.plot(kind='bar',stacked=True)
# Out of curriosity I want to combine these two columns to see the combined correlation

df_survived_grp=df_train.groupby(['Pclass','Sex','Survived']).size().unstack()

df_survived_grp.plot(kind='bar',stacked=True)
# Check survival rates by sibling/spouse numbers

df_survived_grp=df_train.groupby(['SibSp','Survived']).size().unstack()

df_survived_grp.plot(kind='bar',stacked=True)
# Check survival rates by parent/child numbers

df_survived_grp=df_train.groupby(['Parch','Survived']).size().unstack()

df_survived_grp.plot(kind='bar', stacked=True)
# Check the survival rates by the port of embarkment

df_survived_grp=df_train.groupby(['Embarked','Survived']).size().unstack()

df_survived_grp.plot(kind='bar', stacked=True)
# Check the survival by age

df_survived_grp=df_train.groupby(['Age','Survived']).size().unstack()

df_survived_grp
# Create a scatterplot of survival by age

df_train.plot.scatter(x='Age',y='Survived')
# Drop columns that are not useful

df_train.drop('Name',axis=1,inplace=True)

df_train.drop('Ticket',axis=1,inplace=True)

df_train.drop('PassengerId',axis=1,inplace=True)

df_train.head()
# Check again the statistical overview

df_train.describe(include='all')
# Import the required packages

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
# Create a feature matrix

X_le = df_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values

y = df_train[['Survived']].values
# Convert the male/female data to numeric values

le_sex = preprocessing.LabelEncoder()

le_sex.fit(['male','female'])

X_le[:,1] = le_sex.transform(X_le[:,1])
# Convert the embarked data to numeric values

le_embark = preprocessing.LabelEncoder()

le_embark.fit(['C','Q','S','U'])

X_le[:,6] = le_embark.transform(X_le[:,6])
# Check the label-encoded array

X_le
# Create an instance of the classifier

KfoldTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
# Create empty dataframe to store our accuracy numbers

df_accuracy = pd.DataFrame(columns=['Classifier','Accuracy'])
# Get mean accuracy score using the decision tree classifier

accuracy = cross_val_score(KfoldTree, X_le, y, cv=10, scoring='accuracy').mean()

df_accuracy = df_accuracy.append({'Classifier': 'Decision Tree', 'Accuracy': accuracy}, ignore_index=True)



print("Decision Tree Accuracy:")

print(accuracy)
# Import required packages

from sklearn.ensemble import RandomForestClassifier
# Set y as an array

y = np.asarray(df_train['Survived'])

# Create a Gaussian classifier

rfc = RandomForestClassifier(n_estimators=100)
# Get mean accuracy score using the randome forest classifier

# We will reuse the label-encoded features from the decision tree classifier

accuracy = cross_val_score(rfc, X_le, y, cv=10, scoring='accuracy').mean()

df_accuracy = df_accuracy.append({'Classifier': 'Random Forest', 'Accuracy': accuracy}, ignore_index=True)



print("Random Forest Accuracy:")

print(accuracy)
# Import libraries

from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection
# Use one-hot encoding to convert embarked column to numerical values

df_embarked = pd.get_dummies(df_train['Embarked'])

df_embarked.head()
# Use one-hot encoding to convert sex column to numerical values

# Since each column is a perfect predictor for the other, we will drop one

df_sex = pd.get_dummies(df_train['Sex'],drop_first=True)

df_sex.head()
# Combine all dataframes together

df_train_ohe = pd.concat([df_train, df_embarked, df_sex],axis=1)

df_train_ohe.head()
# Create feature matrix and target variable

# Exclude the Sex and Embarked columns

X = np.asarray(df_train_ohe[['Pclass','Age','SibSp','Parch','Fare','C','Q','S','U','male']])

y = np.asarray(df_train_ohe['Survived'])
# Normalize data so smaller variables like Pclass don't get more influence than larger ones like Age

X_norm = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# Set range of values of K to test

k_range = range(1,30)

# Create empty list to store accuracy scores

k_scores = []
# Loop through our range of KNN values to test the model for each

for k in k_range:

    # create the model with the value of K

    knn = KNeighborsClassifier(n_neighbors=k)

    # obtain the cross validation scores

    scores = cross_val_score(knn, X_norm, y, cv=10, scoring='accuracy')

    # append the mean of the scores to our k_scores variable

    k_scores.append(scores.mean())



print(k_scores)
# Plot how the accuracy changes as we change the KNN value of K

import matplotlib.pyplot as plt

%matplotlib inline
# Plot the value of K for KNN versus the cross-validated accuracy

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Accuracy')
# Get mean accuracy score using the KNN classifier with the best results

knn = KNeighborsClassifier(n_neighbors=16)

accuracy = cross_val_score(knn, X_norm, y, cv=10, scoring='accuracy').mean()

df_accuracy = df_accuracy.append({'Classifier': 'KNN', 'Accuracy': accuracy}, ignore_index=True)



print("KNN Accuracy:")

print(accuracy)
# Import library

from sklearn.linear_model import LogisticRegression
# Create feature matrix and target variable

# Logistic Regression requires numerical variables.

# We will reuse the one-hot encoding matrix from above

X = np.asarray(df_train_ohe[['Pclass','Age','SibSp','Parch','Fare','C','Q','S','U','male']])

y = np.asarray(df_train_ohe['Survived'])
# Create instance of logistic regression classifier

LR = LogisticRegression(C=0.01, solver='liblinear')
# Get mean accuracy score using the logistic regression classifier

accuracy = cross_val_score(LR, X, y, cv=10, scoring='accuracy').mean()

df_accuracy = df_accuracy.append({'Classifier': 'Logistic Regression', 'Accuracy': accuracy}, ignore_index=True)



print("Logistic Regression Accuracy:")

print(accuracy)
# Import library

from sklearn.naive_bayes import GaussianNB
# Create a Gaussian classifier

NB = GaussianNB()
# Get mean accuracy score using the naive bayes classifier

# We will reuse the label-encoded features from the decision tree classifier

accuracy = cross_val_score(NB, X_le, y, cv=10, scoring='accuracy').mean()

df_accuracy = df_accuracy.append({'Classifier': 'Naive Bayes', 'Accuracy': accuracy}, ignore_index=True)



print("Naive Bayes Accuracy:")

print(accuracy)
# Since we stored the accuracy of each model in a dataframe, we can easily display them to find which performed the best

df_accuracy.sort_values(by=['Accuracy'], ascending=False)
# Import the test dataset into a new dataframe

csv_path='../input/titanic/test.csv'

df_test = pd.read_csv(csv_path)

df_test.head()
# Check the data types to make sure nothing is improperly typed

df_test.dtypes
# Get a statistical overview of the data

df_test.describe(include='all')
# Drop the same columns as we did for the training dataset

df_test.drop('Name', axis=1, inplace=True)

df_test.drop('Ticket', axis=1, inplace=True)

df_test.drop('Cabin', axis=1, inplace=True)

df_test.head()
# Check if we have any null values

df_test.isnull().sum()
# Use the mean age from our training set to fill in missing ages, rather than calculate from the test data

df_test['Age'].replace(np.nan, mean_age, inplace=True)

df_test['Age'].isnull().sum()
# To fill in missing values for Fare I will calculate the mean value of the test dataset

mean_fare = df_test['Fare'].mean()

mean_fare
# Replace the missing Fare values with the mean

df_test['Fare'].replace(np.nan, mean_fare, inplace=True)

df_test['Fare'].isnull().sum()
# Get a statistical overview of the data after the changes

df_test.describe(include='all')
# Since each sex value is a perfect predictor of the other, we will drop one

df_sex = pd.get_dummies(df_test['Sex'], drop_first=True)

df_sex.head()
# For the emabarked column we can leave all values

df_embarked = pd.get_dummies(df_test['Embarked'])

df_embarked.head()
# Add "U" column to match our training set

df_embarked['U'] = 0

df_embarked.head()
# Combine all dataframes together

df_test_ohe = pd.concat([df_test, df_embarked, df_sex], axis=1)

df_test_ohe.head()
# Create feature matrix

# Exclude the PassengerId, Embarked and Sex columns

X_test = np.asarray(df_test_ohe[['Pclass','Age','SibSp','Parch','Fare','C','Q','S','U','male']])
# Normalize the data

X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
# Fit the KNN model using our full training data set

y = np.asarray(df_train_ohe['Survived'])

knn.fit(X_norm,y)
# Get predictions of our data using the model we fitted above

y_pred = knn.predict(X_test_norm)
# Do a quick check of the results

y_pred
# Create a Survived column in the test dataframe using the predicated values

df_test['Survived'] = y_pred

df_test.head()
# Create a new dataframe of just the column required for the submission

df_submit = df_test[['PassengerId','Survived']]

df_submit.head()
# Export results to a CSV file

csv_predict='predict.csv'

df_submit.to_csv(csv_predict, index=False)