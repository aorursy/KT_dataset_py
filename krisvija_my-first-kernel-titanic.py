import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train2 = train_df.drop(['PassengerId', 'Ticket','Name','Cabin', 'Embarked'], 1)

train2
train2.describe()
pd.isnull(train2).sum() > 0
train2['tran_Age'] = np.log(train2['Age'])

#train2['tran_Age'] = train2['Age']**(-)
sns.jointplot(x='tran_Age', y= 'Survived', data=train2)

train2['Survived'].value_counts()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

train2.iloc[:, 2] = labelencoder.fit_transform(train2.iloc[:, 2])

test2 = test_df.drop(['PassengerId', 'Ticket', 'Name', 'Cabin', 'Embarked'], 1)

test2.iloc[:, 1] = labelencoder.fit_transform(test2.iloc[:, 1])

test2['tran_Age'] = np.log(test2['Age'])

train7 = train2.drop(['Age'],1)

test2 = test2.drop(['Age'],1)

#train2.iloc[:, 7] = labelencoder.fit_transform(train2.iloc[:, 7])

#onehotencoder = OneHotEncoder(categorical_features = [7])

#train2 = onehotencoder.fit_transform(train2).toarray()

train7

test2
from sklearn.preprocessing import Imputer





imputer = Imputer(strategy = 'median', axis = 0, copy = False)

test3 = test_df.drop(['Ticket', 'Name', 'Cabin', 'Embarked', 'Age'],1)

imputer.fit(test2)

test4 = imputer.transform(test2)

imputer.fit(train7)

train3 = imputer.transform(train7)

test4
X_train = train3[:,1:8]

Y_train = train3[:,0]

X_test = test4

X_test
random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)

y_pred = random_forest.predict(X_test)
lr = LogisticRegression()

lr.fit(X_train, Y_train)

y_pred_log = lr.predict(X_test)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = random_forest, X=X_train, y =Y_train, cv = 10)

accuracies
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = lr, X=X_train, y =Y_train, cv = 10)

accuracies
accuracies.mean()
accuracies.std()
from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators' : [10, 100, 500, 1000], 'criterion': ['gini']},

              {'n_estimators' : [10, 100, 500, 1000], 'criterion': ['entropy']}]

grid_search = GridSearchCV(estimator = random_forest,

                          param_grid = parameters,

                          scoring = 'accuracy',

                          cv = 10,

                          n_jobs = -1)

grid_search = grid_search.fit(X_train, Y_train)

best_acc = grid_search.best_score_

best_params = grid_search.best_params_
print (best_acc, best_params)
random_forest = RandomForestClassifier(n_estimators=500, criterion='gini')

random_forest.fit(X_train, Y_train)

y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test3["PassengerId"],

        "Survived": y_pred_log

    })

submission.to_csv('rf_submission.csv', index=False)