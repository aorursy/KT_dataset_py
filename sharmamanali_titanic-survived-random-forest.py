#Importing Libraries

import numpy as np 

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier

#load dataset

dataset = pd.read_csv('../input/train.csv')

dataset_test = pd.read_csv('../input/test.csv') 
#First 5 rows of the dataset

dataset.head()
#statistical analysis of dataset

dataset.describe()
#dataset_1 contains only the numerical values

dataset_1=dataset.drop(['Name','Ticket', 'Cabin','Embarked','Sex','Survived'],axis=1)

dataset_1.head()

#checking for missing values

print('Dataset has null values?')

dataset_1.isnull().values.any()
#Finding missing values in the data set 

total = dataset.isnull().sum()[dataset.isnull().sum() != 0].sort_values(ascending = False)

percent = pd.Series(round(total/len(dataset)*100,2))

pd.concat([total, percent], axis=1, keys=['total_missing', 'percent'])

#Filling in the missing values

dataset_1['Age']=dataset_1['Age'].fillna(dataset_1['Age']).median()

dataset_1['Age'].isnull().values.any()
#Random Forest Machine Learning algortithm applied for estimation

train=dataset_1

test=dataset['Survived']

model=RandomForestClassifier(n_estimators=100)

# training the model...

model.fit(train,test)
#first 5 rows of test dataset

dataset_test.head()
#dataset_1 contains only the numerical values

dataset_test_1=dataset_test.drop(['Name','Ticket', 'Cabin','Embarked','Sex'],axis=1)

dataset_test_1.head()
#Filling in the missing values

dataset_test_1['Age']=dataset_1['Age'].fillna(dataset_test_1['Age']).median()

dataset_test_1['Age'].isnull().values.any()
#filling in all the missing values

dataset_test = dataset_test_1.fillna(dataset_test.mean()).copy()

print(dataset_test)
#predicting

y_pred = model.predict(dataset_test)

y_pred
Submission = pd.DataFrame({ 'PassengerId': dataset_test_1['PassengerId'],

                            'Survived': y_pred })

Submission.to_csv("Submission.csv", index=False)
Submission.head()