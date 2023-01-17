import numpy as np

import pandas as pd

import tensorflow

import seaborn as sns

#train_data = pd.read_csv("train.csv")

#test_data = pd.read_csv("test.csv")

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
#printing first five rows of  train_dataset

train_data.head() 
test_data.head()
# checking shape of train_data

train_data.shape # 891 rows and 12 columns
#checking shape of test data

test_data.shape
train_data.info()
test_data.info()
# sumarie and statistics

train_data.describe()
test_data.describe()
train_data.isnull().sum()
test_data.isnull().sum()
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 

g = sns.heatmap(train_data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True

                , fmt = ".2f", cmap = "coolwarm")
# survival probability

g1 = sns.barplot(x="Sex",y="Survived",data=train_data)

g1 = g.set_ylabel("Survival Probability")
# handle missing value in train_data

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
train_data["Age"].head()
train_data.isnull().sum()
# handle missing value in test_data

test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
test_data.isnull().sum()
# train_data

train_data['Cabin']=train_data['Cabin'].fillna(train_data['Cabin'].mode()[0])

train_data['Embarked']=train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])



train_data.isnull().sum() # all missing values handle
# test_data

test_data['Cabin']=test_data['Cabin'].fillna(test_data['Cabin'].mode()[0])

test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mode()[0])

test_data.isnull().sum()
dataset =  pd.concat([train_data, test_data], axis=0)
dataset.shape
# Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)
# drop name column

dataset.drop(['Name'],axis=1,inplace=True)
dataset.columns
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
encode = dataset[['Sex','Ticket','Cabin','Embarked']].apply(enc.fit_transform)

encode
dataset[['Sex','Ticket','Cabin','Embarked']] = encode[['Sex','Ticket','Cabin','Embarked']]
dataset.head()
dataset.shape
train_len = len(train_data)
train = dataset[:train_len]

test= dataset[train_len:]

test.drop(labels=["Survived"],axis = 1,inplace=True)
test.shape
train.shape
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]



X_train = train.drop(labels = ["Survived"],axis = 1)
from sklearn.ensemble import GradientBoostingClassifier

#model = GradientBoostingClassifier(learning_rate=0.01,max_depth = 2)

model = GradientBoostingClassifier(learning_rate=0.02,max_depth = 2,n_estimators =100)

model.fit(X_train, Y_train)
Score = model.score(X_train, Y_train)

print("Score: %.2f%%" % (Score * 100.0))
predictions = model.predict(test)
output = pd.DataFrame({'Passenger Id': test_data.PassengerId, 'Survived': predictions})
output 
output.to_csv('my_submission1.csv', index=False)

print("Your submission was successfully saved!")