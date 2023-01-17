# Import the necessary Packages

import collections

import numpy as np # linear algebra

from pandas import Series,DataFrame

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load the titanic data set into a dataframe



titanic_training_df = pd.read_csv("../input/train.csv")

titanic_test_df = pd.read_csv("../input/test.csv")



#preview

titanic_training_df.head()
titanic_training_df.info()
titanic_training_df.loc[titanic_training_df["Sex"] == 'male',"Sex"] = 0

titanic_training_df.loc[titanic_training_df["Sex"] == 'female',"Sex"] = 1

titanic_training_df['Embarked'] = titanic_training_df["Embarked"].fillna("S")

titanic_training_df.loc[titanic_training_df["Embarked"] == "S", "Embarked"] = 0

titanic_training_df.loc[titanic_training_df["Embarked"] == "C","Embarked"] = 1

titanic_training_df.loc[titanic_training_df["Embarked"] == "Q","Embarked"] = 2

titanic_training_df.head()
titanic_training_df['Age'] = titanic_training_df["Age"].fillna(titanic_training_df["Age"].median())

#titanic_training_df.info()

titanic_test_df.info()
titanic_test_df.loc[titanic_test_df["Sex"] == 'male',"Sex"] = 0

titanic_test_df.loc[titanic_test_df["Sex"] == 'female',"Sex"] = 1

titanic_test_df["Age"] = titanic_test_df["Age"].fillna(titanic_test_df["Age"].median())

titanic_test_df["Fare"] = titanic_test_df["Fare"].fillna(titanic_test_df["Fare"].median())

titanic_test_df.loc[titanic_test_df["Embarked"] == "S","Embarked"] = 0

titanic_test_df.loc[titanic_test_df["Embarked"] == "C", "Embarked"] = 1

titanic_test_df.loc[titanic_test_df["Embarked"] == "Q","Embarked"] = 2

titanic_test_df.info()
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



alg = LogisticRegression(random_state=1)



#scores = cross_validation.cross_val_score(alg, titanic_training_df[predictors], titanic_training_df["Survived"], cv=3)



#print(scores.mean())

alg.fit(titanic_training_df[predictors],titanic_training_df["Survived"])



predictions = alg.predict(titanic_test_df[predictors])

submission = pd.DataFrame({

        "PassengerId": titanic_test_df["PassengerId"],

        "Survived": predictions

    })



submission.to_csv("titanic_predictions.csv", index=False)

#print(titanic_test_df["PassengerId"],predictions)

#counter = collections.Counter(predictions)

#print(counter.values())



#print(submission)
titanic_training_df['Pclass'].hist()
titanic_training_df.plot.scatter(x='Pclass',y='Survived')