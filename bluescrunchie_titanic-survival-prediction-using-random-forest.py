import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train = train.drop(columns = ['Name'])

test = test.drop(columns = ['Name'])

train.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



# Train set

train['Cabin'] = le.fit_transform(train['Cabin'].astype('str'))

train['Embarked'] = le.fit_transform(train['Embarked'].astype('str'))

train['Ticket'] = le.fit_transform(train['Ticket'].astype('category'))

train['Sex'] = le.fit_transform(train['Sex'].astype('category'))



# Test set

test['Cabin'] = le.fit_transform(test['Cabin'].astype('str'))

test['Embarked'] = le.fit_transform(test['Embarked'].astype('str'))

test['Ticket'] = le.fit_transform(test['Ticket'].astype('category'))

test['Sex'] = le.fit_transform(test['Sex'].astype('category'))



train.head()
from sklearn.impute import SimpleImputer 



train_imputed = pd.DataFrame(SimpleImputer().fit_transform(train))

train_imputed.columns = train.columns



test_imputed = pd.DataFrame(SimpleImputer().fit_transform(test))

test_imputed.columns = test.columns



train = train_imputed

test = test_imputed
X_train = train.copy().drop(columns = ['Survived'])

y_train = train.copy()['Survived']



X_train.head()
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import make_scorer, roc_auc_score



search_space = [

  {

    'max_depth': [60, None],

     'max_features': ['auto', 'sqrt'],

     'min_samples_leaf': [1, 2, 4],

     'min_samples_split': [2, 5, 10],

     'n_estimators': [200, 400, 600, 800, 1000]

  }

]



cv_method = StratifiedKFold(n_splits=5, shuffle = True, random_state=0)

scoring = {'AUC':make_scorer(roc_auc_score)}
from sklearn.ensemble import RandomForestClassifier



optimizer = RandomizedSearchCV(

  estimator = RandomForestClassifier(),

  param_distributions=search_space,

  cv=cv_method,

  scoring=scoring,

  refit='AUC',

  return_train_score = True,

  verbose=1,

  n_iter = 100,

  n_jobs = 10, 

)



rf_classifier = optimizer.fit(X_train, y_train)
features = X_train.columns

imp_dict = {features[i]:optimizer.best_estimator_.feature_importances_[i] for i in range(len(features))}

imp_dict = sorted(imp_dict.items(), key=lambda x: x[1])

print(imp_dict)



plt.bar(*zip(*imp_dict))

plt.xticks(rotation="vertical")

plt.show()
y_pred = rf_classifier.predict(test)

pd.DataFrame(y_pred).head()
submission = pd.DataFrame()

submission['PassengerId'] = test.PassengerId.values.astype('int32')

submission['Survived'] = y_pred.astype('int32')



submission.to_csv('submission.csv',index = False)

submission.head()