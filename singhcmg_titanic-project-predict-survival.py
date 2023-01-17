#Data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import matplotlib.pyplot as plt

%matplotlib inline
#Importing DataSet

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
#Get the Details of DataSet

train_df.info()
#Goal is to make a model which can predict survival

train_df['Survived'].value_counts(normalize = True)
#Now we see there are several Non Numeric values, we have to convert them in to Numeric values

train_df.loc[train_df['Sex'] == 'male', 'Sex'] = 0

train_df.loc[train_df['Sex'] == 'female', 'Sex'] = 1



#We also see there are NAn values in the Dataset, hence we are going to fill the missing data using Mean

train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)
#We are filling all the empty Embarked feature values 

print(train_df["Embarked"].unique())

train_df["Embarked"] = train_df["Embarked"].fillna("S")
#converting all the non numeric values to numeric

train_df.loc[train_df["Embarked"] == "S", "Embarked"] = 0

train_df.loc[train_df["Embarked"] == "C", "Embarked"] = 1

train_df.loc[train_df["Embarked"] == "Q", "Embarked"] = 2
#reviewing the data, ensuing required features are in numeric form

train_df.head()
#Now using Random Forest, we are going to use selective Numeric features and derive Model



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch",

              "Fare", "Embarked"]

X_train, X_test, y_train, y_test = train_test_split(train_df[predictors], train_df["Survived"])
forest = RandomForestClassifier(n_estimators=100,

                                criterion='gini',

                                max_depth=5,

                                min_samples_split=10,

                                min_samples_leaf=5,

                                random_state=0)

forest.fit(X_train, y_train)

print("Random Forest score: {0:.2}".format(forest.score(X_test, y_test)))
test_df.loc[test_df['Sex'] == 'male', 'Sex'] = 0

test_df.loc[test_df['Sex'] == 'female', 'Sex'] = 1



test_df['Age'].fillna(test_df['Age'].mean(), inplace = True)

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace = True)
#now plotting the graps with Features and pick the best feature as good contributor to model



plt.bar(np.arange(len(predictors)), forest.feature_importances_)

plt.xticks(np.arange(len(predictors)), predictors)
#Now making submission in to Prediction



predictors = ["Sex", "Fare", "Pclass", "Age", "SibSp"]

clf = RandomForestClassifier(n_estimators=100,

                             criterion='gini',

                             max_depth=5,

                             min_samples_split=10,

                             min_samples_leaf=5,

                             random_state=0)

clf.fit(train_df[predictors], train_df["Survived"])

prediction = clf.predict(test_df[predictors])



submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": prediction})

submission.to_csv("submission.csv", index=False)