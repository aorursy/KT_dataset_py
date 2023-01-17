import numpy as np 

import pandas as pd 

from IPython.display import display



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
display(train.head())

print(train.info())



PercSurvived = train["Survived"].value_counts(normalize=True)*100

print("\n","Percentage Survived","\n",PercSurvived)
import matplotlib.pyplot as plt

import seaborn as sns



sex_plot = sns.factorplot(x="Sex", y="Survived", data=train,

                   size=5, kind="bar", palette="muted")

sex_plot.set_ylabels("Probability of Survival")

plt.show()
embarked_plot1 = sns.factorplot(x="Embarked", y="Survived", data=train,

                   size=6, kind="bar", palette="muted")

embarked_plot1.set_ylabels("Probability of Survival")



plt.show()

embarked_plot2 = sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

embarked_plot2.set_ylabels("Probability of Survival")



plt.show()
class_plot = sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train, palette="muted")



plt.show()
# Impute missing values for age in training set

train["Age"] = train["Age"].fillna(train["Age"].median())



# Create Child column in training set ('Feature Engineering')

train["Child"] = float("NaN")

train.loc[train["Age"] < 13, "Child"] = 1

train.loc[train["Age"] >= 13, "Child"] = 0



# Create Family Size column for training set

train["Family_Size"] = train["SibSp"] + train["Parch"] + 1



# Simplify Cabin column, by slicing off numbers

# NaN Cabin values labelled as 'N'

train["Cabin"] = train["Cabin"].fillna("N")

train["Cabin"] = train["Cabin"].apply(lambda x: x[0])

# Visualise new features



# Child plot

child_plot = sns.factorplot(x="Child", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

child_plot.set_ylabels("Probability of Survival")



plt.show()



# Famliy_Size plot

family_plot = sns.factorplot(x="Family_Size", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

family_plot.set_ylabels("Probability of Survival")



plt.show()



# Cabin plot

cabin_plot = sns.factorplot(x="Cabin", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

cabin_plot.set_ylabels("Probability of Survival")



plt.show()

# Convert sex to integer values in training set

train.loc[train["Sex"] == "male", "Sex"] = 0

train.loc[train["Sex"] == "female", "Sex"] = 1



# Convert embarked to integer values, and impute missing values

train.loc[train["Embarked"] == "S", "Embarked"] = 0

train.loc[train["Embarked"] == "C", "Embarked"] = 1

train.loc[train["Embarked"] == "Q", "Embarked"] = 2

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].median())



display(train.head())

from sklearn.model_selection import train_test_split



# Create feature and target arrays

X = train.drop(['Survived', 'PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)

y = train['Survived']



# Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=11)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Create random forest classifier

forest = RandomForestClassifier(random_state=11)



# Choose some parameter combinations to try (these values were borrowed from another user)

parameters = {'n_estimators': [4, 6, 9, 100], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

accuracy = make_scorer(accuracy_score)



# Run the grid search with 10-fold cross-validation

grid = GridSearchCV(forest, parameters,scoring=accuracy,cv=10)

grid = grid.fit(X_train, y_train)

print("Tuned Random Forest Parameters: {}".format(grid.best_params_),'\n')

print("Best score is {}".format(grid.best_score_),'\n')



# Set the classifier to the best combination of parameters

forest = grid.best_estimator_



# Fit the best algorithm to the data, and print feature importances & prediction score

forest.fit(X_train, y_train)

print('Feature Importances','\n','Pclass, Sex, Age, Embarked, Child, Family_Size')

print(forest.feature_importances_)



predictions = forest.predict(X_test)

print(accuracy_score(y_test, predictions))



#impute missing values in test

test["Age"] = test["Age"].fillna(test["Age"].median())

#convert to integer values in test

test.loc[test["Sex"] == 'male', 'Sex'] = 0

test.loc[test["Sex"] == 'female', 'Sex'] = 1

test.loc[test["Embarked"] == 'S', 'Embarked'] = 0

test.loc[test["Embarked"] == 'C', 'Embarked'] = 1

test.loc[test["Embarked"] == 'Q', 'Embarked'] = 2

#Create Family Size Column for test

test["Family_Size"] = test["SibSp"] + test["Parch"] + 1

#Create Child column in test set 

test["Child"] = float("NaN")

test.loc[train["Age"] < 13, "Child"] = 1

test.loc[train["Age"] >= 13, "Child"] = 0



display(test.head())
#fit tree to test data

test_features = test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)

prediction = forest.predict(test_features)



#create submission file

forestsubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

   })



forestsubmission.to_csv("forestsubmission.csv",index=False)



display(forestsubmission.head())
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

predictions2 = logreg.predict(X_test)

print('Training Set Score: ',logreg.score(X_train, y_train))



# Print confusion matrix, showing actual numbers of correct and incorrect predictions

print('\n','Confusion Matrix:','\n',confusion_matrix(y_test, predictions2))



# Accuracy on test set (diagonal divided by total in confusion matrix)

print('\n','Test Set Score: ',logreg.score(X_test, y_test))



# Print classification report, showing precision and recall calculated from confusion matrix

# high precision = a low rate of incorrect survival predictions

# high recall = predicted a large number of survivals correctly

print('\n','Classification Report: ','\n',classification_report(y_test, predictions2))

# Fit logreg to Kaggle test data

prediction2 = logreg.predict(test_features)



# Create submission file

logregsubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction2

   })



logregsubmission.to_csv("logregsubmission.csv",index=False)



display(logregsubmission.head())
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

predictions3 = knn.predict(X_test)

print('Training Set Score: ',knn.score(X_train, y_train))

print('Test Set Score: ',knn.score(X_test, y_test))



# Fit knn to kaggle test data

prediction3 = knn.predict(test_features)



# Create submission file

knnsubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction3

   })



knnsubmission.to_csv("knnsubmission.csv",index=False)



display(knnsubmission.head())