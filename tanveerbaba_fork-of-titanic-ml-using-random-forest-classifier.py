# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Our Train data set
train.head()
#Summary of a Dataframe
train.info()
#Generates Descriptive Statistics
train.describe()
#To show missing data I will use Heatmap
plt.figure(figsize = (12,8))
sns.heatmap(train.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)
#Let's first check how people survived in a training set 
sns.countplot(train['Survived'], palette = 'RdBu_r')
#plt.legend(train['Sex'])
#Now we can another property to visualize Survivors along with their class 
sns.countplot(train['Survived'], palette = 'RdBu_r', hue = train['Pclass'])
#Now we can another property to visualize how many were Males and Females
sns.countplot(train['Survived'], palette = 'RdBu_r', hue = train['Parch'])
#Now we can visualize Age of Passengers
plt.figure(figsize =(12,6))
sns.set_style('whitegrid')
sns.distplot(train['Age'].dropna(), kde = False, bins = 30)
#Here we show Passengers count of Siblings and Spouses
plt.figure(figsize = (12,6))
sns.countplot(train['SibSp'], palette = 'viridis')
#Let's Visualize the Embarked Feature
plt.figure(figsize = (12,6))
sns.countplot(train['Embarked'])
#Let's visualize to find out the Average Age's of passengers based on their 'Pclass'
sns.boxplot(x= 'Pclass', y = 'Age', data = train)
#Now let's define a function to fill the 'Age' column's missing fields by the average age of particular classes
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
#call a function to fill the missing 'Age' data
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis = 1)
#Let's visualize the training set again by HeatMap
plt.figure(figsize = (12,6))
sns.heatmap(train.isnull(), cmap = 'viridis', yticklabels = False, cbar = False)
#It's better for us to drop the 'Cabin' column because lot of data is missing in the 'cabin' column
train.drop('Cabin', axis = 1, inplace = True)
#Let's fill in the 'Embarked' Feature
train['Embarked'].fillna(value = 'S', inplace = True)
train.head()
#Now let's drop these features in our train data set
train.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = True)
sex = pd.get_dummies(train['Sex'], drop_first = True)
#Now let's concatenate the sex Dataframe with train Dataframe
train = pd.concat([train,sex], axis = 1)
#Rename the 'male' column and drop the 'Sex' column
train.drop('Sex',axis = 1,inplace = True)
Embarked = pd.get_dummies(train['Embarked'])
#Now let's concatenate the Embarked Dataframe with train Dataframe
train = pd.concat([train,Embarked], axis = 1)
#Drop the 'Embarked' column
train.drop('Embarked',axis = 1,inplace = True)
#Our cleaned Train Data set:
train.head()
#Our Test data set
test.head()
#Summary of a Dataframe 
test.info()
#Generates Descriptive Statistics
test.describe()
#visualize the missing data by heatmap
plt.figure(figsize = (12,8))
sns.heatmap(test.isnull(), cbar = False, yticklabels = False, cmap = 'viridis')
#Let's visualize to find out the Average Age's of passengers based on their 'Pclass'
sns.boxplot(x = 'Pclass', y = 'Age', data = test)
#Now let's define a function to fill the 'Age' column's missing fields by the average age of particular classes
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 27
        else:
            return 24
    else:
        return Age
#call a function to fill the missing 'Age' data
test['Age'] = test[['Age','Pclass']].apply(impute_age, axis = 1)
#visualize the missing data of test data by heatmap
plt.figure(figsize = (12,8))
sns.heatmap(test.isnull(), cbar = False, yticklabels = False, cmap = 'viridis')
#Let's first drop the features we don't need to test for predictions 
PassengerId = pd.DataFrame(test['PassengerId'])
test.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
#Let's visualize to find out the Average Fare's of passengers based on their 'Pclass'
plt.figure(figsize = (10,5))
sns.boxplot(x = 'Pclass', y = 'Fare', data = test)
plt.ylim(0,150)
#Let's Visualize 'Embarked' feature of Test data set
sns.countplot(test['Embarked'])
#Let's fill in the 'Embarked' Feature
test['Embarked'].fillna(value = 'S', inplace = True)
#Now feature 'Fare' has some missing data. Let's try to handle it
def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Fare):
        if Pclass == 1:
            return 60
        elif Pclass == 2:
            return 16
        else:
            return 8
    else:
        return Fare
#call a function to fill the missing 'Fare' data
test['Fare'] = test[['Fare','Pclass']].apply(impute_fare, axis = 1)
sex = pd.get_dummies(test['Sex'], drop_first = True)
#Now let's concatenate the sex Dataframe with test Dataframe
test = pd.concat([test,sex], axis = 1)
#Rename the 'male' column and drop the 'Sex' column
test.drop('Sex',axis = 1,inplace = True)
Embarked = pd.get_dummies(test['Embarked'])
#Now let's concatenate the Embarked Dataframe with test Dataframe
test = pd.concat([test,Embarked], axis = 1)
#Drop the 'Embarked' column
test.drop('Embarked',axis = 1,inplace = True)
#Our cleaned Test Data set:
test.head()
X_train = train.drop('Survived', axis = 1)
y_train = train['Survived']
X_test = test
#Import the model 
from sklearn.ensemble import RandomForestClassifier
#Object creation
rfc = RandomForestClassifier(n_estimators = 151)
#Let's fit the model with the training data set
rfc.fit(X_train,y_train)
#Calculate the predictions
predictions = rfc.predict(X_test)
predictions = pd.DataFrame(predictions)
titanic = pd.concat([PassengerId,predictions], axis = 1)
titanic.columns = ['PassengerId','Survived']
titanic.to_csv('Improved_predictions.csv', index = False)



















