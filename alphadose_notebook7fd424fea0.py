# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



train.head()
del train["Name"]

del test["Name"]

del train["Embarked"]

train["Sex"]=train["Sex"].map({"male":1,"female":0})

test["Sex"]=test["Sex"].map({"male":1,"female":0})



len(train["Cabin"])
train=train.drop(['Ticket',"Cabin"], axis=1)

test=test.drop(['Ticket',"Cabin"],axis=1)
train['Family'] =  train["Parch"] + train["SibSp"]

train['Family'].loc[train['Family'] > 0] = 1

train['Family'].loc[train['Family'] == 0] = 0



test['Family'] =  test["Parch"] + test["SibSp"]

test['Family'].loc[test['Family'] > 0] = 1

test['Family'].loc[test['Family'] == 0] = 0

train = train.drop(['SibSp','Parch'], axis=1)

test    = test.drop(['SibSp','Parch'], axis=1)

for feature in train.columns:

    train[feature]=train[feature].fillna(train[feature].median())
X=train.iloc[:,2:]

y=train.iloc[:,1]

X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV
clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [100], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)
clf.score(X_test,y_test)
ids=test["PassengerId"]
test["Age"]=test["Age"].fillna(test["Age"].median())

test["Fare"]=test["Fare"].fillna(test["Fare"].median())
from sklearn.cross_validation import KFold



def run_kfold(clf):

    kf = KFold(714, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X.values[train_index], X.values[test_index]

        y_train, y_test = y.values[train_index], y.values[test_index]

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)
test=test.drop(["PassengerId","Embarked"],axis=1)
clf.fit(X_train,y_train)

output=clf.predict(test)
result=pd.DataFrame({ 'PassengerId' : ids, 'Survived': output })
result
result.to_csv('titanic-predictions.csv', index = False)