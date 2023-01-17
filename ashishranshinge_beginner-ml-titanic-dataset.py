# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")
titanic_df.describe()
titanic_df.head(5)
titanic_df.info()
#find number of rows with null values in "Embarked" field
titanic_df["Embarked"].isnull().sum()
#replace the null values with the most commonly occurring result
mode_embarked=titanic_df["Embarked"].mode()
mode_embarked.values[0]
titanic_df["Embarked"]=titanic_df["Embarked"].fillna(mode_embarked.values[0])

#Check for nulls
titanic_df["Embarked"].isnull().sum()
#similarly we do this for all the other columns. i.e. fill the missing values with the mode
titanic_df.columns
for null_col in titanic_df.columns:
    mode_of_col = titanic_df[null_col].mode()
    titanic_df[null_col]=titanic_df[null_col].fillna(mode_of_col[0])

for null_col in test_df.columns:
    mode_of_col = test_df[null_col].mode()
    test_df[null_col]=test_df[null_col].fillna(mode_of_col[0])
#check for null values
titanic_df.isnull().any()
test_df.isnull().any()
#find the different levels of Embarked
titanic_df["Embarked"].unique()
titanic_df.columns
test_df.columns
#For regression we need to convert the categorical variables to numerical variables
#data = pd.DataFrame({'color': ['blue', 'green', 'green', 'red']})
#print(pd.get_dummies(data))
titanic_df["Sex"]=(pd.get_dummies(titanic_df["Sex"]))

#similarly we do the same transformations for the test dataset
test_df["Sex"]=(pd.get_dummies(test_df["Sex"]))
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
independent_vars=['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']
logreg.fit(titanic_df[independent_vars],titanic_df['Survived'])
predictions = logreg.predict(test_df[independent_vars])
logreg.intercept_,logreg.coef_
pd.DataFrame(logreg.coef_, columns=independent_vars)
my_prediction = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
my_prediction .head()
filename = 'Titanic_MyPredictions.csv'

my_prediction.to_csv(filename,index=False)

print('Saved file: ' + filename)