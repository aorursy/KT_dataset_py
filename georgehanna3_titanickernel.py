import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train = pd.read_csv("train.csv")

testset = pd.read_csv("test.csv")

testset.info()
print((train.isna().sum()/train.shape[0])*100)
#Drop cabin given it is missing too much data

train_clean = train.drop(['Cabin'],axis=1)

test_clean = test.drop(['Cabin'],axis=1)
train_clean['Age'].fillna(train_clean.groupby('Sex')['Age'].transform("mean"), inplace=True)

test_clean['Age'].fillna(test_clean.groupby('Sex')['Age'].transform("mean"), inplace=True)



train_clean['Embarked'] = train_clean['Embarked'].astype('category').cat.codes

test_clean['Embarked'] = test_clean['Embarked'].astype('category').cat.codes



train_clean['Embarked'].fillna(train_clean.groupby('Pclass')['Embarked'].agg(pd.Series.mode),inplace=True)

test_clean['Embarked'].fillna(test_clean.groupby('Pclass')['Embarked'].agg(pd.Series.mode),inplace=True)
# Convert to categorical values Title 

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in train_clean["Name"]]

train_clean["Title"] = pd.Series(dataset_title)

train_clean["Title"].head()



train_clean["Title"] = train_clean["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_clean["Title"] = train_clean["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

train_clean["Title"] = train_clean["Title"].astype(int)





#Repeat for test set

#Convert to categorical values Title 

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in test_clean["Name"]]

test_clean["Title"] = pd.Series(dataset_title)

test_clean["Title"].head()



test_clean["Title"] = test_clean["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_clean["Title"] = test_clean["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

test_clean["Title"] = test_clean["Title"].astype(int)
train_clean[['C','Q','S']] = pd.get_dummies(train["Embarked"])

test_clean[['C','Q','S']] = pd.get_dummies(test["Embarked"])



train_clean[['M','F']] = pd.get_dummies(train["Sex"])

test_clean[['M','F']] = pd.get_dummies(test["Sex"])



#Combine Sibling Spouse and Parent Child features into one Family Feature

train_clean['FamSize'] = train_clean['SibSp']+train_clean['Parch']+ 1

test_clean['FamSize'] = test_clean['SibSp']+test_clean['Parch']+ 1



#Drop Original Feature columns now that we've encoded them

train_clean = train_clean.drop(['Embarked','Sex','Ticket','SibSp','Parch','PassengerId','Name'],axis=1)

test_clean = test_clean.drop(['Embarked','Sex','Ticket','SibSp','Parch','PassengerId','Name'],axis=1)



train_clean.describe()
from sklearn.preprocessing import RobustScaler

train_clean[["Age","Fare"]] = RobustScaler().fit_transform(train_clean[["Age","Fare"]])

test_clean[["Age","Fare"]] = RobustScaler().fit_transform(test_clean[["Age","Fare"]])
g= sns.pairplot(train_clean, hue= "Survived")
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(train_clean.corr(), vmax=1.0, center=0, fmt='.2f', square=True, linewidths=.5, annot=True, cbar_kws={"shrink":.70})
y_train = train_clean["Survived"].values

X_train = train_clean.drop(columns=["Survived"])
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection



models = []

models.append(('RF',RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)))

models.append(('NB',GaussianNB()))

models.append(("SVM",SVC(gamma='auto')))

models.append(("LR",LogisticRegression(solver='lbfgs')))



results = []

names = []



for name, model in models:

    kfold = model_selection.KFold(n_splits =10, random_state = 7)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring ="accuracy")

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
from sklearn.model_selection import RandomizedSearchCV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



print(rf_random.best_params_)

best_model = rf_random.best_estimator_

preds = best_model.predict(X_train)

accuracy_score(y_train, preds)
from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [int(x) for x in np.linspace(10, 30, num = 10)],

    'max_features': [2,3],

    'min_samples_leaf': [1],

    'min_samples_split': [10],

    'n_estimators': [int(x) for x in np.linspace(1400, 1800, num = 30)]

}



# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

best_model = grid_search.best_estimator_

preds = best_model.predict(X_train)

accuracy_score(y_train, preds)