# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#data analysis libraries 



import numpy as np

import pandas as pd





# visualization libraries



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
test.head()
sum(train.duplicated(subset = 'PassengerId')) == 0

sum(test.duplicated(subset = 'PassengerId')) == 0
print (train.isnull().values.any())

print (test.isnull().values.any())
test.shape
train.shape
train.info()
test.info()
train.describe(include="all")
#Null values - train dataset



print(pd.isnull(train).sum())
#check the null % of coulmns



round(100*(train.isnull().sum())/len(train.index))
#Null values - test dataset



print(pd.isnull(test).sum()) 
#check the null % of coulmns



round(100*(test.isnull().sum())/len(test.index))
train.drop(['Cabin'], axis=1, inplace=True)

test.drop(['Cabin'], axis=1, inplace=True)
train.drop(['Ticket'], axis=1, inplace=True)

test.drop(['Ticket'], axis=1, inplace=True)
# Let's see the correlation matrix 

plt.figure(figsize = (10,8))        



# Size of the figure

sns.heatmap(train.corr(),annot = True)

plt.show()
# We will visualize various features to understand their relations.



sns.barplot(x="Pclass", y="Survived", data=train)
# Percentage of Pclass = 1 who survived



train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100
print(pd.isnull(train).sum()) 
train = train.fillna({"Embarked": "S"})
sns.barplot(x="Embarked", y="Survived", data=train)
# Percentage of Embarked = C who survived



train["Survived"][train["Embarked"] == 'C'].value_counts(normalize = True)[1]*100
#Percentage of females who survived



train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100
#Percentage of males who survived



train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100
sns.barplot(x="Sex", y="Survived", data=train)
sns.barplot(x="SibSp", y="Survived", data=train)
sns.barplot(x="Parch", y="Survived", data=train)

train["Age"] = train["Age"].fillna(29.69)

test["Age"] = test["Age"].fillna(29.69)



train["Age_Group"] = pd.cut(train.Age,[-1, 5, 12, 18, 35, 60, np.inf], labels=['Baby', 'Child', 'Teenager', 'Youngster', 'Adult', 'Senior Citizen'])

test["Age_Group"] = pd.cut(test.Age,[-1, 5, 12, 18, 35, 60, np.inf], labels=['Baby', 'Child', 'Teenager', 'Youngster', 'Adult', 'Senior Citizen'])

round(100*(train.isnull().sum())/len(train.index))
sns.barplot(x="Age_Group", y="Survived", data=train)
test["Fare"] = test["Fare"].fillna(32.20)



train["Fare_Range"] = pd.cut(train.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])

test["Fare_Range"] = pd.cut(test.Fare,[-1, 130, 260, 390, 520], labels=['1', '2', '3', '4'])
sns.barplot(x="Fare_Range", y="Survived", data=train)
train.drop(['Fare', 'Age', 'Name'], axis=1, inplace=True)

test.drop(['Fare', 'Age', 'Name'], axis=1, inplace=True)
#We will map Sex, Embarked & Age_Group to numerical value



age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Youngster': 4, 'Adult': 5, 'Senior Citizen': 6}

train['Age_Group'] = train['Age_Group'].map(age_mapping)

test['Age_Group'] = test['Age_Group'].map(age_mapping)
sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

test['Embarked'] = test['Embarked'].map(embarked_mapping)

train['Embarked'] = train['Embarked'].map(embarked_mapping)

Fare_Range_mapping = {"1": 1, "2": 2, "3": 3, "4": 4}

test['Fare_Range'] = test['Fare_Range'].map(Fare_Range_mapping)

train['Fare_Range'] = train['Fare_Range'].map(Fare_Range_mapping)



train.head()
# Logistic Regression





from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

columns = ['Pclass', 'Sex', 'SibSp','Embarked', 'Age_Group', 'Fare_Range']



from sklearn.model_selection import train_test_split



test_df = test

X = train[columns]

y = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=0)



from sklearn.metrics import accuracy_score

lr.fit(X_train,y_train)

predictions = lr.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(accuracy)
# Cross validation



from sklearn.model_selection import cross_val_score



lr = LogisticRegression()

scores = cross_val_score(lr, X, y, cv=10)

accuracy = np.mean(scores)

print(scores)

print(accuracy)
# final model



columns = ['Pclass', 'Sex', 'SibSp','Embarked', 'Age_Group', 'Fare_Range']

lr = LogisticRegression()

lr.fit(X,y)

test_df_predictions = lr.predict(test_df[columns])
# Create a Submission dataframe





test_df_ids = test_df["PassengerId"]

submission_df = {"PassengerId": test_df_ids,

                 "Survived": test_df_predictions}

submission = pd.DataFrame(submission_df)

#submission.head()
# Create a submission file for Kaggle

submission.to_csv("submission.csv",index=False)