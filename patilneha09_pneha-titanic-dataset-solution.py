# Import all the necessary packages

import numpy as np 

import pandas as pd

import statsmodels.api as sms

from sklearn.linear_model import LogisticRegression

import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from random import randint

# suppress warnings from final output

import warnings

warnings.simplefilter("ignore")
# Load test dataset

test_df=pd.read_csv('../input/test.csv')

test_df.head()
# Load train dataset

train_df=pd.read_csv('../input/train.csv')

train_df.head()
train_df.info()
test_df.info()
train_df.sample(5)
train_df.Name.value_counts()
# Combine datasets to reduce repetition of code

single=[train_df,test_df]
# Remove the maiden names specified for married women

for data in single:

    data['Name']=data['Name'].apply(lambda x: list(x.split('('))[0])
# Remove the special characters with nicknames

for data in single:

    data['Name']=data['Name'].apply(lambda x: list(x.split('"'))[0])
train_df['Name'].value_counts()
# Calculate mean of age in each class for train dataset

train_df.groupby(['Pclass'])['Age'].describe()
# Generate random numbers between the lowest and highest mean

train_df['Age']=train_df['Age'].fillna(randint(24,37))
# Calculate mean of age in each class for test dataset

test_df.groupby(['Pclass'])['Age'].describe()
# Generate random numbers between the lowest and highest mean

test_df['Age']=test_df['Age'].fillna(randint(24,42))
# Find the most frequent 'Embarked' value

freq_embark=train_df.Embarked.mode()[0]

freq_embark
# Fill the missing values

train_df['Embarked']=train_df['Embarked'].fillna(freq_embark)
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].median())
#Test whether the above code worked

train_df.info()
#Test whether the above code worked

test_df.info()
# Map columns to numerical values

for data in single:

    data['Sex']=data['Sex'].map({'female':1,'male':0}).astype(int)

    data['Embarked']=data['Embarked'].map({'S':1,'C':2,'Q':3}).astype(int)
train_df.head()
train_df.info()
train_df['intercept']=1

model=sms.Logit(train_df['Survived'],train_df[['intercept','Pclass','Sex','Age','SibSp','Parch','Embarked']])

result=model.fit()

result.summary()
g=sns.FacetGrid(data=train_df,col='Survived',row='Pclass');

g.map(plt.hist,'Age');
train_df.groupby(['Pclass','Sex','Survived'])['Survived'].count()
g=sns.FacetGrid(data=train_df,col='Survived',row='Pclass');

g.map(sns.countplot,'Sex');
train_df.drop(columns=['PassengerId','Name','Parch','Ticket','Fare','Cabin','Embarked','intercept'],inplace=True)
train_df.info()
test_df.drop(columns=['Name','Parch','Ticket','Fare','Cabin','Embarked'],inplace=True)
test_df.info()
# Diivide the data into test and train

X_train=train_df.drop('Survived',axis=1)

Y_train=train_df['Survived']

X_test=test_df.drop('PassengerId',axis=1)
# Fit the model

model = LogisticRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Calucalte the accuracy

accuracy = round(model.score(X_train, Y_train), 4)

accuracy
# Gather the solution into a new dataframe

final_df = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })
final_df.head()
final_df.shape
# Save the dataframe to a csv file

final_df.to_csv('submission.csv',index=False)