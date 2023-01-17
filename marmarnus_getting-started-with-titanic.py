# This is my model that I presented on 12 Febuary.

# I used Alexis Cook's tutorial as a baseline for creating the model whereafter I added the Age and Fare variables as features.

# You can also use this as a starting point for and improve upon this model - See if you can beat my score of 0.77990!

# You can try a different model - like sklearn's LogiscticRegression or try adding/removing some features.





#Load packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

#Using missingno - identify columns containing missing values.

msno.matrix(train_data)
#Using missingno - create barplot showing ratio of missing values to other. 

msno.bar(train_data)
#Using missingno - identify relationships between variables with missing values.

msno.dendrogram(train_data)
#Passenger class - confirm that Passenger class is a categorical variable.

train_data.Pclass
#Convert catergorical variable into dummy variables.

pd.get_dummies(train_data.Pclass)
#Encode categorical variables.

train_data["Pclass"] = pd.get_dummies(train_data["Pclass"])

train_data["Sex"] = pd.get_dummies(train_data["Sex"])

train_data["SibSp"] = pd.get_dummies(train_data["SibSp"])

train_data["Parch"] = pd.get_dummies(train_data["Parch"])



test_data["Pclass"] = pd.get_dummies(test_data["Pclass"])

test_data["Sex"] = pd.get_dummies(test_data["Sex"])

test_data["SibSp"] = pd.get_dummies(test_data["SibSp"])

test_data["Parch"] = pd.get_dummies(test_data["Parch"])
#Import seaborn graphing package

import seaborn as sns



#Calculate correlations between variables in train_data 

correlation = train_data.corr()



#display correlations as a heatmap

sns.heatmap(train_data.corr())
#Select features from train_data

features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
#Fill missing values in Age column.

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
#Check for missing values in features of train_data

#If there are any missing values a true/1 will be return, otherwise false/0.

train_data[features].isna().head()
#Sum over the columns 

#if the sum of any column is greater than 0 it contains a missing value

train_data[features].isna().sum()
#Repeat for test_data: Check for missing values in features of test_data

test_data[features].isna().sum()
#Check for other missing values in features

msno.matrix(test_data[features])
#Replace missing value in Fare column with average of the column

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

#Make sure it worked

test_data[features].isna().sum()
from sklearn.ensemble import RandomForestClassifier

#from sklearn.linear_model import LogisticRegression



#Create target for training

y_train = train_data["Survived"]



#Select only the features from the datasets

X_train = train_data[features]

X_test = test_data[features]



#Create the model

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

#model = LogisticRegression()



#Fit the model

model.fit(X_train, y_train)



#Use the model to create predictions on the test dataset

predictions = model.predict(X_test)



#Create output csv file

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")