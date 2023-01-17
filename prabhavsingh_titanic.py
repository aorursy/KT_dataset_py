import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

acc = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
del train['Name']

del train['PassengerId']

del train['Ticket']
del train['Cabin']



def CleanEmbarked(str):

    if str == 'S':

        return 0

    elif str == 'C':

        return 1

    else:

        return 2

    

train['Port'] = train['Embarked'].apply(CleanEmbarked)

del train['Embarked']



def CleanSex(str):

    if str == 'male':

        return 0

    else:

        return 1

    

train['Gender'] = train['Sex'].apply(CleanSex)

del train['Sex']
train.describe()
train['Age'].fillna(0, inplace=True)
train.describe()
Y_train = train['Survived']

del train['Survived']
X_train = np.array(train.values)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.shape)

print(Y_train.shape)
del test['Name']

del test['PassengerId']

del test['Ticket']



del test['Cabin']



def CleanEmbarked(str):

    if str == 'S':

        return 0

    elif str == 'C':

        return 1

    else:

        return 2

    

test['Port'] = test['Embarked'].apply(CleanEmbarked)

del test['Embarked']



def CleanSex(str):

    if str == 'male':

        return 0

    else:

        return 1

    

test['Gender'] = test['Sex'].apply(CleanSex)

del test['Sex']



test['Age'].fillna(0, inplace=True)
test['Fare'].fillna(0, inplace=True)
X_test = np.array(test.values)
X_test_scaled = scaler.transform(X_test)
from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_train_scaled, Y_train, random_state = 0)
clf1 = SVC()

grid = {'C' : [1, 10, 1000, 10000], 'kernel' : ['linear', 'rbf'], 'gamma' : [0.1, 0.01, 0.001]}

CV = GridSearchCV(clf1, grid)
clf_SVC = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)



clf_SVC.fit(X_train_1, Y_train_1)

Y_pred_SVC = clf_SVC.predict(X_test_1)

print(classification_report(Y_test_1, Y_pred_SVC))

print(clf_SVC.score(X_test_1, Y_test_1))
clf2 = RandomForestClassifier()

grid2 = {'n_estimators' : [5, 10, 30], 'criterion' : ['gini', 'entropy'], 'max_depth' : [1, 10]}

CV2 = GridSearchCV(clf2, grid2)
CV2.fit(X_train_1, Y_train_1)

print(CV2.best_estimator_)
clf_RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

                       max_depth=10, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=30,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)



clf_RF.fit(X_train_1, Y_train_1)

Y_pred_RF = clf_RF.predict(X_test_1)

print(classification_report(Y_test_1, Y_pred_RF))

print(clf_RF.score(X_test_1, Y_test_1))
predictions = clf_SVC.predict(X_test_scaled)

print(predictions)
submission = pd.DataFrame({'PassengerId':acc['PassengerId'],'Survived':predictions})
submission.head()
filename = 'Titanic Predictions 2.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)