import pandas as pd

import numpy as np
# Read data

# Test - to be predicted

test = pd.read_csv('../input/titanic/test.csv')

# Train - training data

train = pd.read_csv('../input/titanic/train.csv')
# get PassengerId from test columns. This will help in prediction later

testPassengerIds = test['PassengerId']

testPassengerIds.head()
# Drop some columns

train.drop(['PassengerId', 'Name', 'Ticket'], inplace = True, axis = 1)

test.drop(['PassengerId', 'Name', 'Ticket'], inplace = True, axis = 1)
train.head()
test.head()
train.isnull().any()
test.isnull().any()
train['Cabin'].unique()
# Dictionary for mappinig

# Fill each place with First letter to label mapping



train['Cabin'].fillna(0, inplace = True)

def getCabin(value):

    val_dict = {

        'A' : 6,

        'B' : 5,

        'C' : 4,

        'D' : 3,

        'E' : 2,

        'F' : 1,

        'T' : 1   ## Taking T same as F, taking it to be an error     

    }

    return val_dict.get(str(value)[0], 0)



train['Cabin'] = train["Cabin"].apply(getCabin)

test['Cabin'] = test['Cabin'].apply(getCabin)
train.head()
# Fill with most common values

train['Embarked'].fillna(train['Embarked'].mode().item() , inplace = True)

# To ensure no discrepancy

test['Embarked'].fillna(train['Embarked'].mode().item(), inplace = True)
import plotly.express as px

fig = px.sunburst(train, path=['Embarked', 'Pclass', 'Sex'], values='Survived', title = 'Embarked -> Class -> Sex')

fig.show()
# One hot encode the Data

train = pd.get_dummies(train, drop_first=True)

test = pd.get_dummies(test, drop_first=True)
# Impute age with median

mean = train['Age'].median()

train['Age'].fillna(mean, inplace = True)

test['Age'].fillna(mean, inplace = True)
# using `|` makes or operator, checks if missing in train or test

train.isnull().any() | test.isnull().any()
# Impute fare with mean of training data

meanFare = train['Fare'].mean()

test['Fare'].fillna(meanFare, inplace = True)
train.head()
test.head()
# Put the log of fare and class

def correctedLog(value):  # So that we do not get infinity

    return np.log(1 + value)



train['Status'] = train['Pclass'] + train['Fare'].apply(correctedLog)

test['Status'] = test['Pclass'] + test['Fare'].apply(correctedLog)
train['RootAgeTimesClass'] = train['Age'].apply(np.sqrt) * train['Pclass']

test['RootAgeTimesClass'] = test['Age'].apply(np.sqrt) * test['Pclass']
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train.head()
train['Young'] = train['Age'] <= train['Age'].mean()

test['Young'] = test["Age"] <= train['Age'].mean()
train['YoungMale'] = train['Young'] & train['Sex_male']

test['YoungMale'] = test["Young"] & test['Sex_male']
train.head()
X = train.iloc[:, 1: ]

y = train.iloc[:, 0]

y.shape, X.shape
import matplotlib.pyplot as plt

plt.hist(train['Fare']);

# Visualising the Box Cox Transform

from scipy.stats import boxcox

xt,_ = boxcox(train['Fare'] + 1)

plt.hist(xt);
# Apply Box Cox Transformation

train['Fare'], maxlog = boxcox(train['Fare'] + 1)

test['Fare'] = boxcox(test['Fare'] + 1, lmbda = maxlog)
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state =55, test_size = 0.2, shuffle = True)

y_train.shape, y_val.shape
X_train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

np.random.seed(0)
# Parameters for Grid Search

paramDict = {

    'n_estimators' : [5, 10, 25, 50, 75, 100, 200, 500],

    'max_depth' : [4, 8, 10, 15, 20, 50],

    

}

# Random Forest Model

model = RandomForestClassifier(n_jobs = 8)

# Grid Search CV

clf = GridSearchCV(estimator=model, param_grid=paramDict, n_jobs=10)
clf.fit(X_train, y_train)
clf.best_params_, clf.best_score_
f1_score(clf.predict(X_val), y_val)
# Make model with best parameters, fit with all data now

finalModel = RandomForestClassifier(**clf.best_params_)



# Fit Data

finalModel.fit(X, y)



# Generate Predictions

y_preds = finalModel.predict(test)



#########################################################################

# Submission File Generation

file_name = "Submission_16_08_6.csv"



y_pred_series = pd.Series(y_preds.flatten(), name = 'Survived')



file = pd.concat([testPassengerIds, y_pred_series], axis = 1)



file.to_csv(file_name, index = False);