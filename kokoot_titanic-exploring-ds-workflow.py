# import needed packages

import math

import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import ParameterGrid

import sklearn.metrics as metrics

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# load datasets

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.describe()
# check for NaN values

train.isnull().sum()
# make a copy of our dataset

train1 = train.copy()



# drop columns

train1.drop(columns=['Name', 'Cabin', 'PassengerId'], inplace=True)
# drop rows with NaN

train1.dropna(inplace=True)
# convert string columns to type int

# we will put this into a function since we will be using this later on

def string_to_int(dataset):

    string_cols = dataset.select_dtypes(['object']).columns

    dataset[string_cols].nunique()

    dataset[string_cols] = dataset[string_cols].astype('category').apply(lambda x: x.cat.codes)

    

# call function

string_to_int(train1)
train1
# splitting dataset into training and validation data

Xdata = train1.drop(columns = ['Survived'])

ydata = train1['Survived']

rd_seed = 333 # data splitted randomly with this seed

Xtrain, Xval, ytrain, yval = train_test_split(Xdata, ydata, test_size=0.25, random_state=rd_seed)
# chosen range of hyperparameters for Decision Tree Classifier

param_grid = {

    'max_depth': range(1,100), 

    'criterion': ['entropy', 'gini']

}

# using sklearn's ParameterGrid to make a grid of parameter combinations

param_comb = ParameterGrid(param_grid)
# iterate each parameter combination and fit model

val_acc = []

train_acc = []

for params in param_comb:

    dt = DecisionTreeClassifier(**params)

    dt.fit(Xtrain, ytrain)

    train_acc.append(metrics.accuracy_score(ytrain, dt.predict(Xtrain)))

    val_acc.append(metrics.accuracy_score(yval, dt.predict(Xval)))
# plotting how accurate our model is with each hyperparameter combination

plt.figure(figsize=(20,6))

plt.plot(train_acc,'or-')

plt.plot(val_acc,'ob-')

plt.xlabel('hyperparametr index')

plt.ylabel('accuracy')

plt.legend(['train', 'validation'])

best_params = param_comb[np.argmax(val_acc)]

best_params
print(param_comb[5]) # printing hyperparameters with index 5

print(param_comb[105]) # printing hyperparameters with index 105
# chosen range of hyperparameters for Random Forest Classifier

param_grid = {

    'max_depth': range(1, 30), 

    'n_estimators': range(1, 500, 25),

}

# using sklearn's ParameterGrid

param_comb = ParameterGrid(param_grid)
# iterate each parameter combination and fit model

val_acc = []

train_acc = []

for params in param_comb:

    dt = RandomForestClassifier(**params)

    dt.fit(Xtrain, ytrain)

    train_acc.append(metrics.accuracy_score(ytrain, dt.predict(Xtrain)))

    val_acc.append(metrics.accuracy_score(yval, dt.predict(Xval)))
# plotting how accurate our model is with each hyperparameter combination

plt.figure(figsize=(20,6))

plt.plot(train_acc,'or-')

plt.plot(val_acc,'ob-')

plt.xlabel('hyperparametr index')

plt.ylabel('accuracy')

plt.legend(['train', 'validation'])

best_params = param_comb[np.argmax(val_acc)]

best_params
print(param_comb[0]) # printing hyperparameters with index 0

print(param_comb[1]) # printing hyperparameters with index 1

print(param_comb[200]) # printing hyperparameters with index 1
# chosen range of hyperparameters for AdaBoost Classifier

param_grid = {

    'n_estimators': range(1, 1000, 25),

    'algorithm': ['SAMME', 'SAMME.R'],

}

# using sklearn's ParameterGrid

param_comb = ParameterGrid(param_grid)
# iterate each parameter combination and fit model

val_acc = []

train_acc = []

for params in param_comb:

    dt = AdaBoostClassifier(**params)

    dt.fit(Xtrain, ytrain)

    train_acc.append(metrics.accuracy_score(ytrain, dt.predict(Xtrain)))

    val_acc.append(metrics.accuracy_score(yval, dt.predict(Xval)))
# plotting how accurate our model is with each hyperparameter combination

plt.figure(figsize=(20,6))

plt.plot(train_acc,'or-')

plt.plot(val_acc,'ob-')

plt.xlabel('hyperparametr index')

plt.ylabel('accuracy')

plt.legend(['train', 'validation'])

best_params = param_comb[np.argmax(val_acc)]

best_params
print(param_comb[0]) # printing hyperparameters with index 0

print(param_comb[50]) # printing hyperparameters with index 1

print(param_comb[75]) # printing hyperparameters with index 75
def rank_classifiers(data):

    rd_seed = 333

    # specify the data we want to predict

    Xdata = data.drop(columns = ['Survived'])

    ydata = data['Survived']

    # ready Classifiers and their distinctive range of parameters for

    # hyperparameter tuning with cross-validation using GridSearchCV

    pipeline = Pipeline([

        ('clf', DecisionTreeClassifier()) # placeholder classifier

    ])

    # narrowed down hyperparameters

    parameters = [

        {

            'clf': (DecisionTreeClassifier(),),

            'clf__max_depth': range(1, 15), 

            'clf__criterion': ['gini', 'entropy'],

        }, {

            'clf': (RandomForestClassifier(),),

            'clf__max_depth': range(1, 10), 

            'clf__n_estimators': range(25, 500, 25),

        }, {

            'clf': (AdaBoostClassifier(),),

            'clf__n_estimators': range(1, 250, 10),

            'clf__algorithm': ['SAMME', 'SAMME.R'],

        }

    ]

    # run GridSearchCV with training data to determine the best 

    # classifier with the best parameters while cross-validating them

    # thus maximizing fairness of score rankings

    clf = GridSearchCV(pipeline, parameters, cv=5, iid=False, n_jobs=-1)

    clf.fit(Xdata, ydata)

    print('accuracy score (train): {0:.6f}'.format(clf.best_score_))

    # now lets see how well it predicts our testing data

    return clf
clf = rank_classifiers(train1)
print(clf.best_params_)
# print sum of missing values for each column

for dataset in [train, test]:

    print(dataset.isna().sum())
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}



for dataset in [train ,test]:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    # we can now drop the cabin feature

    dataset.drop(columns=['Cabin'], inplace=True)
for dataset in [train, test]:

    mean = dataset["Age"].mean()

    std = dataset["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # create an array of random numbers given the range and length

    rand_ages_for_nan = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in the column age with random values generated

    ages_slice = dataset["Age"].copy()

    ages_slice[np.isnan(ages_slice)] = rand_ages_for_nan

    dataset["Age"] = ages_slice.astype(int)
# describe data to find the most frequent one

for dataset in [train, test]:

    print(dataset.Embarked.describe())
# fill missing values with the frequent value

for dataset in [train, test]:

    dataset.Embarked.fillna('S', inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)
# final check of missing values

for dataset in [train, test]:

    print(dataset.isna().sum())
train.drop(columns=['PassengerId'], inplace=True)
for dataset in [train, test]:

    dataset['Title'] = dataset.Name.str.extract(' ([A-z]+\.)', expand=False)

    # print added Titles

    print(dataset.Title.value_counts())

for dataset in [train, test]:

    dataset.drop(columns=['Name'], inplace=True)
for dataset in [train, test]:

    # create relatives feature by summing up sibsp and parch

    dataset['Relatives'] = dataset.SibSp + dataset.Parch

    # create not_alone feature, where it is 1 if # of relatives is 0

    dataset.loc[dataset['Relatives'] > 0, 'Not_alone'] = 0

    dataset.loc[dataset['Relatives'] == 0, 'Not_alone'] = 1
for dataset in [train, test]:

    string_to_int(dataset)
train.describe()
test.describe()
for dataset in [train, test]:

    dataset.loc[dataset.Age > 65, 'Age_cat'] = 6

    dataset.loc[dataset.Age <= 65, 'Age_cat'] = 5

    dataset.loc[dataset.Age <= 48, 'Age_cat'] = 4

    dataset.loc[dataset.Age <= 35, 'Age_cat'] = 3

    dataset.loc[dataset.Age <= 24, 'Age_cat'] = 2

    dataset.loc[dataset.Age <= 13, 'Age_cat'] = 1

    dataset.loc[dataset.Age <= 1, 'Age_cat'] = 0

    # drop age column

    dataset.drop(columns=['Age'], inplace=True)
for dataset in [train, test]:

    dataset.Fare = pd.qcut(dataset.Fare, 6, labels=[0,1,2,3,4,5]).astype(int)
for dataset in [train, test]:

    dataset.drop(columns=['Ticket'], inplace=True)
train.describe()
test.describe()
plt.subplots(figsize = (15,10))

sns.heatmap(train.corr(), annot=True,cmap="RdYlGn_r")

plt.title("Feature Correlations", fontsize = 18)
clf = rank_classifiers(train)
print(clf.best_params_)
importances = pd.DataFrame({'feature':train.drop(columns=['Survived']).columns,'importance':np.round(clf.best_params_['clf'].feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)

importances.plot.bar()
# extract PassengerId

id_col = test.PassengerId

# predict survival

test = test.drop(columns=['PassengerId'])
# create submission dataframe

submission = pd.DataFrame({

    'PassengerId': id_col.values,

    'Survived': clf.predict(test)

})
# save submission

submission.to_csv('submission.csv', index=False)