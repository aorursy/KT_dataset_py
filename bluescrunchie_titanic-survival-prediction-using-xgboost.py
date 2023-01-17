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
X_train = train.copy().drop(columns = ['Survived'])

y_train = train.copy()['Survived']



X_train.head()
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import make_scorer, roc_auc_score



search_space = [

  {

    'n_estimators': [100, 200, 300, 400],

    'learning_rate': [0.01, 0.1, 0.2, 0.3],

    'max_depth': range(3, 10),

    'colsample_bytree': [i/10.0 for i in range(1, 3)],

    'gamma': [i/10.0 for i in range(3)],

  }

]



cv_method = StratifiedKFold(n_splits=5, shuffle = True, random_state=0)

scoring = {'AUC':make_scorer(roc_auc_score)}
from xgboost import XGBClassifier



optimizer = RandomizedSearchCV(

  estimator = XGBClassifier(),

  param_distributions=search_space,

  cv=cv_method,

  scoring=scoring,

  refit='AUC',

  return_train_score = True,

  verbose=1,

  n_iter = 100,

  n_jobs = 10, 

)



xgb_classifier = optimizer.fit(X_train, y_train)
features = X_train.columns

imp_dict = {features[i]:optimizer.best_estimator_.feature_importances_[i] for i in range(len(features))}

imp_dict = sorted(imp_dict.items(), key=lambda x: x[1])

print(imp_dict)



plt.bar(*zip(*imp_dict))

plt.xticks(rotation="vertical")

plt.show()
y_pred = xgb_classifier.predict(test)

pd.DataFrame(y_pred).head()
submission = pd.DataFrame()

submission['PassengerId'] = test.PassengerId.values

submission['Survived'] = y_pred



submission.to_csv('submission.csv',index = False)

submission.head()