import pandas as pd

import numpy as np



train = pd.read_csv('../input/train.csv')

train.head()
train.info()
train.describe()
train['Gender'] = train['Sex'].map({'male': 1, 'female': 0}).astype(int)
train.head()
if len(train.Embarked[ train.Embarked.isnull() ]) > 0:

    train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train['Embarked']))) 

Ports_dict = { name : i for i, name in Ports }

train.Embarked = train.Embarked.map(Ports_dict)
train.head(6)
median_age = train['Age'].dropna().median()

if len(train.Age[ train.Age.isnull() ]) > 0:

    train.loc[ (train.Age.isnull()), 'Age'] = median_age
train.head(6)
# define X and y

feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender']

X = train[feature_cols]

y = train.Survived
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# make class predictions for the testing set

from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=100)

forest.fit(X_train, y_train)

y_pred_class = forest.predict(X_test)
# check the classification accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
# evaluate with 5-fold cross-validation (using X instead of X_train)

from sklearn.cross_validation import cross_val_score

cross_val_score(forest, X, y, cv=5, scoring='accuracy').mean()
test = pd.read_csv('../input/test.csv')

test.head()
# I need to do the same with the test data now, so that the columns are the same as the training data

# I need to convert all strings to integer classifiers:

# female = 0, Male = 1

test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# Embarked from 'C', 'Q', 'S'

# All missing Embarked -> just make them embark from most common place

if len(test.Embarked[ test.Embarked.isnull() ]) > 0:

    test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values

# Again convert all Embarked strings to int

test.Embarked = test.Embarked.map(Ports_dict)





# All the ages with no data -> make the median of all Ages

median_age = test['Age'].dropna().median()

if len(test.Age[ test.Age.isnull() ]) > 0:

    test.loc[ (test.Age.isnull()), 'Age'] = median_age



# All the missing Fares -> assume median of their respective class

if len(test.Fare[ test.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = test[ test.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1 ), 'Fare'] = median_fare[f]



# define X and y

X_test = test[feature_cols]
test.head()
# make class predictions for the testing set

forest.fit(X, y)

test_pred_class_forest = forest.predict(X_test)
# create a DataFrame that only contains the IDs and predicted classes for the test data

pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':test_pred_class_forest}).set_index('PassengerId').head()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X, y)

test_pred_class_logreg = logreg.predict(X_test)



# create a submission file 

pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':test_pred_class_logreg}).set_index('PassengerId').head()