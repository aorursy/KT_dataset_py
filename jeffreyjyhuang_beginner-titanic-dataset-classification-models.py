# Libraries

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



# Import data

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

passID = test_data['PassengerId']



# Print number of rows and columns

print(f"Shape of Train Data: {train_data.shape}")

print(f"Shape of Test Data: {test_data.shape}")
train_data.head()
# Missing Values in TRAIN

print("Missing Values in Train Data")

for column in train_data.columns:

    mask = train_data[column].isna().sum()

    if mask != 0:

        print(f"Column `{column}` has {mask} missing values, representing {round(mask/train_data.shape[1], 2)}% of the data")

print("")

print("Missing Values in Test Data")

# Missing Values in TEST

for column in test_data.columns:

    mask = train_data[column].isna().sum()

    if mask != 0:

        print(f"Column `{column}` has {mask} missing values, representing {round(mask/train_data.shape[1], 2)}% of the data")
# Fill two missing values in `Embarked` with S (majority)

train_data['Embarked'] = train_data['Embarked'].fillna("S")

test_data['Embarked'] = train_data['Embarked'].fillna("S")
# Fill age with medianand change into integer values

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median()).apply(int)

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median()).apply(int)
# Drop non-numeric columns (Name, Ticket), columns with too many NA values (Cabin), and correlated columns (Fare)

train_data = train_data.drop(columns = ['Cabin', 'Name', 'Ticket', 'Fare'])

test_data = test_data.drop(columns = ['Cabin', 'Name', 'Ticket', 'Fare'])
# Adding sibling and parent column

train_data['Family'] = train_data['SibSp'] + train_data['Parch'] + 1

test_data['Family'] = test_data['SibSp'] + test_data['Parch'] + 1
train_data['Sex:Male'] = pd.get_dummies(train_data['Sex'])['male']

test_data['Sex:Male'] = pd.get_dummies(test_data['Sex'])['male']
# Adding dummified columns

train_data['Embarked:C'] = pd.get_dummies(train_data['Embarked'])['C']

train_data['Embarked:S'] = pd.get_dummies(train_data['Embarked'])['S']

# Same but to test data

test_data['Embarked:C'] = pd.get_dummies(test_data['Embarked'])['C']

test_data['Embarked:S'] = pd.get_dummies(test_data['Embarked'])['S']
columns_to_drop = ['Sex', 'SibSp', 'Parch', 'Embarked']

train_data = train_data.drop(columns = columns_to_drop)

test_data = test_data.drop(columns = columns_to_drop)
train_data
# Split X & y

from sklearn.model_selection import train_test_split

X = train_data.drop(columns = ['Survived'])

y = train_data['Survived']



# Train/Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

correlation = X_train.corr()

plt.figure(figsize = (8,8))

sns.heatmap(correlation, square = True, cmap = 'mako').set(title='Correlation of Training Data Variables')
# Scaling Data

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

test_data = scaler.transform(test_data)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
# Fitting Logistic Regression

log_reg_model = LogisticRegression()

log_reg_model.fit(X_train, y_train)





# Scoring

train_prediction = log_reg_model.predict(X_train)

test_prediction = log_reg_model.predict(X_test)

accuracy_train = accuracy_score(train_prediction, y_train)

accuracy_test = accuracy_score(test_prediction, y_test)



print(f"Score on training set: {accuracy_train}")

print(f"Score on test set: {accuracy_test}")
from sklearn.tree import DecisionTreeClassifier



# lists for scoring

training_scores = []

test_scores = []



# finding optimal depth

for i in range(1,15):

    decision_tree = DecisionTreeClassifier(max_depth=i)

    decision_tree.fit(X_train, y_train)

    training_scores.append(decision_tree.score(X_train,y_train)*100)

    test_scores.append(decision_tree.score(X_test,y_test)*100)



# Plotting training and test scores against max_depth = i

plt.figure()



plt.plot(training_scores, label = 'Training Scores')

plt.plot(test_scores, label = 'Test Scores')

plt.xticks(range(1,15,1))

plt.title('Relationship between training score and test score vs max_depth')

plt.xlabel('Maximum Depth')

plt.ylabel('Score')

plt.legend()



plt.show()



# Print best test score:

highest_score = max(test_scores)

highest_index = test_scores.index(highest_score)



print(f"Best Accuracy Score: Max Depth of {highest_index + 1}")

print(f"Training accuracy of {round(training_scores[highest_index], 2)}% (max_depth = {highest_index + 1})")

print(f"Test accuracy of {round(test_scores[highest_index],2)}% (max_depth = {highest_index + 1})")
decision_tree_3 = DecisionTreeClassifier(max_depth=3)

decision_tree_3.fit(X_train, y_train)

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier





AB_model = AdaBoostClassifier()

AB_model.fit(X_train, y_train)



grad_boost_model = GradientBoostingClassifier()

grad_boost_model.fit(X_train, y_train)



XGB_model = XGBClassifier(

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1)

XGB_model.fit(X_train, y_train)





print("=="*20)

print(f"AdaBoost Training score: {AB_model.score(X_train,y_train)}")

print(f"AdaBoost Testing score: {AB_model.score(X_test,y_test)}")

print("=="*20)

print(f"Gradient Boosting Training score: {grad_boost_model.score(X_train,y_train)}")

print(f"Gradient Boosting Testing score: {grad_boost_model.score(X_test,y_test)}")

print("=="*20)

print(f"XGBoost Training score: {XGB_model.score(X_train,y_train)}")

print(f"XGBoost Testing score: {XGB_model.score(X_test,y_test)}")

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier





AB_model = AdaBoostClassifier()

AB_model.fit(X_train, y_train)



grad_boost_model = GradientBoostingClassifier()

grad_boost_model.fit(X_train, y_train)



XGB_model = XGBClassifier()

XGB_model.fit(X_train, y_train)



print("=="*20)

print(f"AdaBoost Training score: {AB_model.score(X_train,y_train)}")

print(f"AdaBoost Testing score: {AB_model.score(X_test,y_test)}")

print("=="*20)

print(f"Gradient Boosting Training score: {grad_boost_model.score(X_train,y_train)}")

print(f"Gradient Boosting Testing score: {grad_boost_model.score(X_test,y_test)}")

print("=="*20)

print(f"XGBoost Training score: {XGB_model.score(X_train,y_train)}")

print(f"XGBoost Testing score: {XGB_model.score(X_test,y_test)}")

pred = decision_tree_3.predict(test_data)

submission_dict = {"PassengerId": passID, "Survived": pred }

submission = pd.DataFrame(submission_dict)

submission
submission.to_csv("titanic_submission.csv", index = False)
pred = XGB_model.predict(test_data)

submission_dict = {"PassengerId": passID, "Survived": pred }

xg_submission = pd.DataFrame(submission_dict)

xg_submission.to_csv("titanic_submission_xg.csv", index = False)