# import pandas, and numpy
import pandas as pd
import numpy as np
from pandas import Series,DataFrame

# matplotlib for visualization
import matplotlib.pyplot as plot
import seaborn as sns

# ignore warnings for now
import warnings
warnings.filterwarnings('ignore')
# load the training set
train = pd.read_csv('../input/train.csv')

# load the test set
test = pd.read_csv('../input/test.csv')

# print some samples from the training set
train.sample(5)
# describe training data
train.describe(include='all')
sns.barplot(x="Embarked", y="Survived", data=train)

print("Number of passengers who embarked at S: ", (train["Embarked"] == "S").value_counts(normalize = True)[1] *100)
print("Number of passengers who embarked at C: ", (train["Embarked"] == "C").value_counts(normalize = True)[1] *100)
print("Number of passengers who embarked at Q: ", (train["Embarked"] == "Q").value_counts(normalize = True)[1] *100)
sns.barplot(x="Pclass", y="Survived", data=train)
sns.barplot(x="Sex", y="Survived", data=train)
sns.barplot(x="SibSp", y="Survived", data=train)
sns.barplot(x="Parch", y="Survived", data=train)
train['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract('([A-Za-z]+)\.', expand=False)

sns.barplot(x="Title", y="Survived", data=train)
train = train.drop(['PassengerId', 'Ticket'], axis=1)
test = test.drop('Ticket', axis=1)

test.describe(include='all')
# fill NaN values with S, since S is the most occured value
train["Embarked"] = train["Embarked"].fillna("S")
train.describe(include='all')
# fill in the missing fare values in test set
# use the median value to fill up missing rows
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.describe(include='all')
# get average, std and missing values for age in training set
training_age_avg = train["Age"].mean()
training_age_std = train["Age"].std()
training_age_missing = train["Age"].isnull().sum()

print("Avg Training Age:", training_age_avg, "Std Training Age:", training_age_std, 
      "Missing Age:",training_age_missing)

# get average, std and missing values for age in test set
test_age_avg = test["Age"].mean()
test_age_std = test["Age"].std()
test_age_missing = test["Age"].isnull().sum()

print("Avg Test Age:", test_age_avg, "Std Test Age:", test_age_std, "Missing Age:",test_age_missing)

# generate random number between (mean - std) & (mean + std)
random_1 = np.random.randint(training_age_avg - training_age_std, training_age_avg + training_age_std,
                            size = training_age_missing)

random_2 = np.random.randint(test_age_avg - test_age_std, test_age_avg + test_age_std, size = test_age_missing)

train['Age'].dropna()
train["Age"][np.isnan(train["Age"])] = random_1

test['Age'].dropna()
test["Age"][np.isnan(test["Age"])] = random_2
combine = [train, test]

# integer mapping for male and female vars
sex_map = {"male": 0, "female": 1}

# map male and female to integers
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map(sex_map)

# extract out male and female as separate features in the dataset
for dataset in combine:
    dataset['Male'] = dataset['Sex'].map(lambda s: 1 if s == 0 else 0)
    dataset['Female'] = dataset['Sex'].map(lambda s: 1 if  s == 1  else 0)

# remove Sex feature, as we already have male and female feature
train = train.drop('Sex', axis=1)
test = test.drop('Sex', axis=1)

train.head()
combine = [train, test]

# integer mapping for embarked feature
embark_map ={"S": 0, "C": 1, "Q": 2}

# map S, C and Q to integers
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].map(embark_map)

# extract out S, C and Q as separate features in the dataset
for dataset in combine:
    dataset['Embarked_S'] = dataset['Embarked'].map(lambda s: 1 if s == 0 else 0)
    dataset['Embarked_C'] = dataset['Embarked'].map(lambda s: 1 if  s == 1  else 0)
    dataset['Embarked_Q'] = dataset['Embarked'].map(lambda s: 1 if  s == 2  else 0)

# remove Embarked feature, as we already have S, C and Q feature
train = train.drop('Embarked', axis=1)
test = test.drop('Embarked', axis=1)

train.head()
combine = [train, test]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

# extract out different titles as features
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title_Mr'] = dataset['Title'].map(lambda s: 1 if s == 1 else 0)
    dataset['Title_Miss'] = dataset['Title'].map(lambda s: 1 if  s == 2  else 0)
    dataset['Title_Mrs'] = dataset['Title'].map(lambda s: 1 if  s == 3  else 0)
    dataset['Title_Master'] = dataset['Title'].map(lambda s: 1 if  s == 4  else 0)
    dataset['Title_Royal'] = dataset['Title'].map(lambda s: 1 if  s == 5  else 0)
    dataset['Title_Rare'] = dataset['Title'].map(lambda s: 1 if  s == 6  else 0)

# remove Title feature, as we already have different titles as feature
train = train.drop(['Title', 'Name'], axis=1)
test = test.drop(['Title', 'Name'], axis=1)

train.head()
combine = [train, test]

# create a Fsize feature
for dataset in combine:
    dataset["Fsize"] = dataset["SibSp"] + train["Parch"] + 1

# extract out Fsize into 4 different features
for dataset in combine:
    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

# remove Fsize, SibSp and Parch feature, as we already have features on family size
train = train.drop(['Fsize', 'SibSp', 'Parch'], axis=1)
test = test.drop(['Fsize', 'SibSp', 'Parch'], axis=1)

train.head()
combine = [train, test]

# extract out Pclass into 3 different features
for dataset in combine:
    dataset['Upper_Class'] = dataset['Pclass'].map(lambda s: 1 if s == 1 else 0)
    dataset['Middle_Class'] = dataset['Pclass'].map(lambda s: 1 if  s == 2  else 0)
    dataset['Lower_Class'] = dataset['Pclass'].map(lambda s: 1 if s == 3 else 0)

# remove Pclass, as we already have features on classes
train = train.drop('Pclass', axis=1)
test = test.drop('Pclass', axis=1)

train.head()
combine = [train, test]

for dataset in combine:
    dataset['FareBand'] = pd.qcut(dataset['Fare'], 4, labels = [1, 2, 3, 4])

train = train.drop('Fare', axis=1)
test = test.drop('Fare', axis=1)

train.head()
combine = [train, test]

# fill missing cabin values with U (undefined)
for dataset in combine:
    dataset["Cabin"] = dataset["Cabin"].fillna('U')

# integer mapping for cabins
cabin_map = {"U": 0, "C": 1, "E": 2, "G": 3, "D": 4, "A": 5, "B": 6, "F": 7, "T": 8}

# map integers to cabin
for dataset in combine:
    dataset["Cabin"] = dataset["Cabin"].map(cabin_map)

for dataset in combine:
    dataset['Cabin_U'] = dataset['Cabin'].map(lambda s: 1 if s == 0 else 0)
    dataset['Cabin_C'] = dataset['Cabin'].map(lambda s: 1 if  s == 1  else 0)
    dataset['Cabin_E'] = dataset['Cabin'].map(lambda s: 1 if s == 2 else 0)
    dataset['Cabin_G'] = dataset['Cabin'].map(lambda s: 1 if s == 3 else 0)
    dataset['Cabin_D'] = dataset['Cabin'].map(lambda s: 1 if s == 4 else 0)
    dataset['Cabin_A'] = dataset['Cabin'].map(lambda s: 1 if s == 5 else 0)
    dataset['Cabin_B'] = dataset['Cabin'].map(lambda s: 1 if s == 6 else 0)
    dataset['Cabin_F'] = dataset['Cabin'].map(lambda s: 1 if s == 7 else 0)
    dataset['Cabin_T'] = dataset['Cabin'].map(lambda s: 1 if s == 8 else 0)

train = train.drop(['Cabin'], axis=1)
test = test.drop('Cabin', axis=1)
X_train = train
Y_train = train["Survived"]
X_test = test.copy()

X_test.head()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
lr.score(X_train, Y_train)
from sklearn import svm

svm = svm.SVC()
svm.fit(X_train, Y_train)
Y_pred = svm.predict(X_test)
svm.score(X_train, Y_train)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

kfold = StratifiedKFold(n_splits=10)

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_

ids = test["PassengerId"]

predictions = gsadaDTC.predict(X_test)

ids = test["PassengerId"]
int_id = []
for i in ids:
    int_id.append(int(i))
    
int_pred = []
for y in predictions:
    int_pred.append(int(y))

output = pd.DataFrame({"PassengerId": int_id, "Survived": int_pred})
output.to_csv("submission.csv", index=False)

gsadaDTC.best_score_
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
Y_pred = gbc.predict(X_test)
gbc.score(X_train, Y_train)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=10)

RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [n for n in range(9, 14)],
              "max_features": [1, 3, 10],
              "min_samples_split": [n for n in range(4, 11)],
              "min_samples_leaf": [n for n in range(2, 5)],
              "bootstrap": [False],
              "n_estimators" :[n for n in range(10, 60, 10)],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
predictions = gsRFC.predict(X_test)
gsRFC.score(X_train, Y_train)

ids = test["PassengerId"]
int_id = []
for i in ids:
    int_id.append(int(i))
    
int_pred = []
for y in predictions:
    int_pred.append(int(y))

output = pd.DataFrame({"PassengerId": int_id, "Survived": int_pred})
output.to_csv("submission.csv", index=False)


gsRFC.best_score_

# print ("Starting 1")
# forrest_params = dict(     
#     max_depth = [n for n in range(9, 14)],     
#     min_samples_split = [n for n in range(4, 11)], 
#     min_samples_leaf = [n for n in range(2, 5)],     
#     n_estimators = [n for n in range(10, 60, 10)],
# )
# print ("Starting 2")

# forrest = RandomForestClassifier()
# print ("Starting 3")

# forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 
# print ("Starting 4")

# forest_cv.fit(X_train, Y_train)
# print ("Starting 5")

# print("Best score: {}".format(forest_cv.best_score_))
# print("Optimal params: {}".format(forest_cv.best_estimator_))

# # random forrest prediction on test set
# predictions = forest_cv.predict(X_test)
# print ("Starting 6")

# output = pd.DataFrame({"PassengerId": ids, "Survived": predictions})
# output.to_csv("submission.csv", index=False)

# print ("Starting 7")


# print("Best score: {}".format(forest_cv.best_score_))
# print("Optimal params: {}".format(forest_cv.best_estimator_))