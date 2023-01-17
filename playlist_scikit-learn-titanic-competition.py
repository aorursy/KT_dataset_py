import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from time import time

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
dataset = [train, test]

for data in dataset:

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    ms = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    ms = ms[ms["Percent"] > 0]

    print(ms)

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='90')

    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)
## Data Manipulations

## NaN

data = [train, test]

for dataset in data:

    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())

    dataset["Embarked"] = dataset["Embarked"].fillna(0)

    ## Binarize Male / Female

    dataset["Sex"].loc[dataset["Sex"] == "male"] = 0

    dataset["Sex"].loc[dataset["Sex"] == "female"] = 1

    #Create 'Child' feature

    dataset["IsChild"] = 0

    dataset["IsChild"].loc[dataset["Age"] < 18] = 1

    dataset["IsChild"].loc[dataset["Age"] > 18] = 0

    #Numerize 'Embarked feature'

    dataset["Embarked"].loc[dataset["Embarked"] == "S"] = 1

    dataset["Embarked"].loc[dataset["Embarked"] == "C"] = 2

    dataset["Embarked"].loc[dataset["Embarked"] == "Q"] = 3



    ## Create Title Class

    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

    ## Create Age Group class

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
test['Fare'].fillna(test['Fare'].median(), inplace = True)
for dataset in data:

    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare','Average_fare','high_fare'])
train = train.drop(['Name', 'Cabin', 'Fare', 'Ticket', 'PassengerId'], axis=1)

test2 = test

test = test.drop(['Name', 'Cabin', 'Fare', 'Ticket', 'PassengerId'], axis=1)
train = pd.get_dummies(train, columns = ["Fare_bin"], prefix=["Fare_type"])

test = pd.get_dummies(test, columns = ["Fare_bin"], prefix=["Fare_type"])
print("Head Train")

train.head()
print("Head test")

test.head()
print('check the nan value in train data')

print(train.isnull().sum())

print('___'*30)

print('check the nan value in test data')

print(test.isnull().sum())
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
## SPLITTING THE DATA

from sklearn.model_selection import train_test_split



all_X = train.drop('Survived', axis=1)

all_y = train['Survived']



train_X, test_X, train_y, test_y = train_test_split(

    all_X, all_y, test_size=0.2,random_state=0)

print("Data correctly splitted")
## CHECK FOR 'NaN'

print('check the nan value in train data')

print(train.isnull().sum())

print('___'*30)

print('check the nan value in test data')

print(test.isnull().sum())
from sklearn import tree



tree_clf = tree.DecisionTreeClassifier(min_samples_split=15)

t0 = time()

print("Training...")

tree_clf.fit(train_X, train_y)

print("Training Completed in:", round(time()-t0, 2), "s")



pred = (tree_clf.predict(test_X))

acc = accuracy_score(test_y, pred)

print("ACCURACY:", round(acc*100, 2), "%")
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=6500, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

t0 = time()

print("Training...")

forest_clf.fit(train_X, train_y)

print("Training completed in:", round(time()-t0, 2), "s")



pred = (forest_clf.predict(test_X))

acc = accuracy_score(test_y, pred)

print("ACCURACY:", round(acc*100, 2), "%")



print("Important features")

pd.Series(forest_clf.feature_importances_,train_X.columns).sort_values(ascending=True).plot.barh(width=0.8)

print('__'*30)

print(acc)
from sklearn.neural_network import MLPClassifier



nnc_clf = MLPClassifier(

    hidden_layer_sizes=(100, ),

    activation='relu',

    solver='adam',

    alpha=0.0001,

    batch_size='auto',

    learning_rate='constant',

    learning_rate_init=0.001,

    power_t=0.5, max_iter=500,

    shuffle=True, random_state=None,

    tol=0.0001,

    verbose=False,

    warm_start=False,

    momentum=0.9,

    nesterovs_momentum=True,

    early_stopping=False,

    validation_fraction=0.1,

    beta_1=0.9,

    beta_2=0.999, 

    epsilon=1e-08,

    n_iter_no_change=10)



t0 = time()

print("Training...")

nnc_clf.fit(train_X, train_y)

print("Training completed in:", round(time()-t0, 2), "s")



nnc_clf.predict(test_X)

acc = nnc_clf.score(test_X, test_y)

print("ACCURACY:", round(acc*100, 2), "%")
from sklearn.model_selection import cross_val_score

clf = [forest_clf, nnc_clf, tree_clf]

for clf in clf:

    t0 = time()

    print(clf)

    scores = cross_val_score(clf, all_X, all_y, cv=10)

    print("Validation completed in:", round(time()-t0, 2), "s")

    print(scores)

    print("MOYENNE:", np.mean(scores)*100, "%")

    print("---"*30, "\n")
from sklearn.model_selection import GridSearchCV

# Random Forest Classifier Parameters tunning 

model = RandomForestClassifier()

n_estim=range(100,1000,100)



## Search grid for optimal parameters

param_grid = {"n_estimators" :n_estim}





model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



model_rf.fit(train_X,train_y)







# Best score

print(model_rf.best_score_)



#best estimator

model_rf.best_estimator_
selectedClf = forest_clf



selectedClf.fit(all_X, all_y)

pred = selectedClf.predict(test)

pred
## Export results

holdout_ids = test2["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": pred}

submission = pd.DataFrame(submission_df)



submission.to_csv('titanic_submission.csv', index=False)