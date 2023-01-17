# This Python 3 environment has in-built analytic libraries installed that we will use
# The definition for the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra/statistical calculations
import pandas as pd # data processing
# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# From sklearn, these are the models we will be looking at
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
import os
print(os.getcwd())
print(os.listdir("../input")) #list the files in the input directory

### Load the train and test data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

print('-----train dataset column types-----')
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Variable", " Data Type"]
print(dtype_df)
# Could also get same information (in messier format) with:
# print (train_df.info())

# preview first five records of the data
train_df.head()
# look at quartile summaries of each variable
train_df.describe()
#print('----test dataset column types information-------')
print (test_df.info())

# preview first five records of the data
test_df.head()
nans = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nans[nans.sum(axis=1) > 0])
# Age: fill with random values ranging from (mean-stand dev, mean+stand dev)
train_random_ages = np.random.randint(train_df["Age"].mean() - train_df["Age"].std(),
                                          train_df["Age"].mean() + train_df["Age"].std(),
                                          size = train_df["Age"].isnull().sum())

test_random_ages = np.random.randint(test_df["Age"].mean() - test_df["Age"].std(),
                                          test_df["Age"].mean() + test_df["Age"].std(),
                                          size = test_df["Age"].isnull().sum())

# replace the nan in the Age columns with a random value
train_df["Age"][np.isnan(train_df["Age"])] = train_random_ages
test_df["Age"][np.isnan(test_df["Age"])] = test_random_ages
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)
# Embarked: replace nan's with S, the most common
train_df["Embarked"].fillna('S', inplace=True)
test_df["Embarked"].fillna('S', inplace=True)

# Fare: test_df has one missing value; we can also replace nans with the mode(2 vals) or the median (1 value)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
# Look at Class of passengers 
print(train_df[['Pclass', 'Survived']]
      .groupby(['Pclass'], as_index=False).mean())
print("_"*50)

# Look at Sex of passengers to confirm the assumptions
print (train_df[["Sex", "Survived"]]
       .groupby(['Sex'], as_index=False).mean())
print("_"*50)

# For the ages, lets look at graphs
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Because passengers may be contained in families, we may want to look families as a single unit
# We can do this by looking at Parch and SibSp
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
      .sort_values(by='Survived', ascending=False))
print("_"*50)
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
      .sort_values(by='Survived', ascending=False))
# Fare may have some information, as the richer people had a higher Survived rate
#  However, similar to Age, we may want to bucket Fare values into ranges

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
corr = train_df.corr()
corr.style.background_gradient()
# Drop the Ticket, Cabin, PassengerId as we will not use them
train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
# Create FamilySize combining the SibSp and the Parch
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
#Look at FamilySize correlation
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Create FamilySizeGroup bucket: Alone: 1 person, Small: 2-4, and Big: 5 or more 
# Create IsAlone feature
for dataset in combine:
    dataset['FamilySizeGroup'] = 'Small'
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Big'
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
# map the feature to be ordinal
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}
for dataset in combine:
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)

# drop the features FamilySize, Parch, SibSp as we no longer need them
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
# Extract Title from the Name using regular expressions (ie the ' ([A-Za-z]+)\.')
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Compress titles into basic categories (reduces the number of categories for the model)
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# Convert the categorical into ordinal values for modelling
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Drop the Name feature now
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.head()
train_df['AgeBucket'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBucket', 'Survived']].groupby(['AgeBucket'], as_index=False).mean().sort_values(by='AgeBucket', ascending=True)
# Replace the Age with ordinals based on the buckets
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df = train_df.drop(['AgeBucket'], axis=1)
combine = [train_df, test_df]
train_df.head()
train_df['FareBucket'] = pd.qcut(train_df['Fare'], 4, duplicates='drop')
train_df[['FareBucket', 'Survived']].groupby(['FareBucket'], as_index=False).mean().sort_values(by='FareBucket', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    

train_df = train_df.drop(['FareBucket'], axis=1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train_df.head()
test_df.head()
# Prep work
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
# Show the shapes of each table
X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 5)
acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 5)
acc_knn
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 5)
acc_svc
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 5)
acc_gaussian
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 5)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 5)
acc_random_forest
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)
Y_pred = xgb.predict(X_test)
acc_xgb = round(xgb.score(X_train, Y_train) * 100, 5)
acc_xgb
### How to select tune the hyperparameters of the model (such as reducing the depth of the tree)
### to find the optimal xgboost model
# import xgboost as xgb
# from sklearn.model_selection import RandomizedSearchCV

# X_dummies = pd.get_dummies(train_df, drop_first= True)
# X = X_dummies[:len(train_df)]
# y = train_df.Survived

# # Create the parameter grid: gbm_param_grid 
# gbm_param_grid = {
#     'n_estimators': range(8, 20),
#     'max_depth': range(6, 10),
#     'learning_rate': [.4, .45, .5, .55, .6],
#     'colsample_bytree': [.6, .7, .8, .9, 1]
# }

# # Instantiate the regressor: gbm
# gbm = XGBClassifier(n_estimators=10)

# # Perform random search: grid_mse
# xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
#                                     estimator = gbm, scoring = "accuracy", 
#                                     verbose = 1, n_iter = 50, cv = 4)

# # Fit randomized_mse to the data
# xgb_random.fit(X, y)

# # Print the best parameters and lowest RMSE
# print("Best parameters found: ", xgb_random.best_params_)
# print("Best accuracy found: ", xgb_random.best_score_)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree', 'XGBoost'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_decision_tree, acc_xgb]})
models.sort_values(by='Score', ascending=False)
# The random forest has the highest score, so we will be using that for our predictions
Y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('titanic.csv', header = True, index = False)