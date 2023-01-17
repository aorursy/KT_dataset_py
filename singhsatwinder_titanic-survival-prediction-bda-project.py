import numpy as np # linear algebra

import pandas as pd # data processing , for importing csv files

import matplotlib.pyplot as plt  # data visualization

import seaborn as sns   # data visualization



import os

print(os.listdir("../input"))

# importing the dataset

train_data=pd.read_csv("../input/train.csv")
train_data.head()
# descriptive Statistics

train_data.describe()
train_data[['Pclass','Survived']].groupby(['Pclass']).mean()
train_data[['Sex','Survived']].groupby(['Sex']).mean()
# checking survival with Sex

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train_data,hue='Sex' )
# checking survival with our Pclass

sns.countplot(x='Survived',data=train_data,hue='Pclass')
train_data['Cabin'].isnull().value_counts()
# droping name, ticket and Cabin column from table

train_data.drop(['Name','Ticket','Cabin'],inplace=True,axis=1)
# checking number of passengers with unknown Age

train_data['Age'].isnull().value_counts()
# checking for null values in data set

sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)
sns.barplot(x='Pclass',y='Age',data=train_data)
# function to fill unknown ages according to class of passenger

def age_fill(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age
# applying age_fill() function to data set

train_data['Age']=train_data[['Age','Pclass']].apply(age_fill,axis=1)
# checking data set

train_data.head(10)
# converting Categorial variable Sex to binary variable.

train_data['Sex']=train_data['Sex'].apply(lambda x : 1 if x=='male' else 0 )
# converting Categorial variable Embarked to two binary variable.

# pd.get_dummies returns data table with number of columns according to number of categories.



Embarked=pd.get_dummies(train_data['Embarked'],drop_first=True)
# removing Embarked column from data set

train_data.drop('Embarked',axis=1,inplace=True)
# adding binary colums from Embarked cloumn to our data set

train_data=pd.concat([train_data,Embarked],axis=1)
# checking the head of data set

train_data.head()
# used logistic regression 

from sklearn.linear_model import LogisticRegression
model=LogisticRegression() 
# here X is the input

# y is output corresponding to input in X.

X=data.drop('Survived',axis=1)

y=data['Survived']
# fitting data to our model

model.fit(X,y)
# importing testing data

test_data=pd.read_csv('../input/test.csv')
test_data.head()
# cleaning test data as we have done for training data

test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
test_data['Sex']=test_data['Sex'].apply(lambda x : 1 if x=='male' else 0 )
sns.heatmap(test_data.isnull(),cbar=False)
test_data['Age']=test_data[['Age','Pclass']].apply(age_fill,axis=1)
test_data.head()
Embarked=pd.get_dummies(test_data['Embarked'],drop_first=True)  
test_data.drop('Embarked',axis=1,inplace=True)
test_data=pd.concat([test_data,Embarked],axis=1)
test_data["Fare"]=test_data["Fare"].fillna(value=test_data['Fare'].mean()) 
# calculating the predictions

predictions=model.predict(test_data)
# creating table for result

passen_id=test_data['PassengerId']
predict_=pd.Series(data=predictions)
result=pd.concat([passen_id,predict_],axis=1)
result.to_csv('my_submission')
result.head(10)