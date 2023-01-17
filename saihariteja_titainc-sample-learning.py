

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer

titainc_df=pd.read_csv('../input/train.csv')
titainctest_df = pd.read_csv('../input/test.csv')
titainc_df.head()
#this helps in getting top 5 rows
titainctest_df.head()
titainc_df.describe()
#Here we have missing values only in Age rest we dont have any missing values
sns.countplot(x="Sex",data=titainc_df)
# From the below graph it is evidant that we have more male passengers
#Now we will check how many male and female passengers survived
sns.countplot(x="Sex",data=titainc_df,hue="Survived")
titainc_df.groupby('Embarked')['Survived'].mean().plot(kind='barh')
titainc_df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
#Let us start building a model.
titainc_df["Sex"] = titainc_df["Sex"].apply(lambda sex: 0 if sex == 'male' else 1)
y = targets = labels = titainc_df["Survived"].values
columns = ["Fare", "Pclass", "Sex", "Age", "SibSp"]
features = titainc_df[list(columns)].values
features
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
X = imp.fit_transform(features)
X
#Import models from scikit learn module let us try using Decision tree and Random Forest for modelling:
from sklearn import tree
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one = my_tree_one.fit(X, y)
print(my_tree_one.feature_importances_) 
print(my_tree_one.score(X, y))
titainctest_df["Sex"] = titainctest_df["Sex"].apply(lambda sex: 0 if sex == 'male' else 1)
#features_test = train_df[list(columns)].values
features_test = titainctest_df[list(columns)].values
imp_test = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_test = imp_test.fit_transform(features_test)
X_test
pred = my_tree_one.predict(X_test)
pred
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(X, y)

#Print the score of the new decison tree
print(my_tree_two.score(X, y))
# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(X, y)
# Print the score of the fitted random forest
print(my_forest.score(X, y))

pred = my_forest.predict(X_test)
pred
my_submission = pd.DataFrame({'PassengerID': titainctest_df.PassengerId, 'Survived': pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)