import pandas as pd

import numpy as np

from numpy import nan
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.head(3)
train.shape, test.shape
train.info()
train.head()
test.head()
train.isnull().sum()/len(train)
test.isnull().sum()/len(test)
[train[i].unique() for i in train.iloc[:,[1,2,4,6,7,11]].columns]
train.drop(['PassengerId'], axis=1, inplace=True)

test.drop(['PassengerId'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])

test['Sex'] = le.fit_transform(test['Sex'])
train['Embarked'].value_counts()
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Deck'] = train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

test['Deck'] = test['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
train['FSize'] = train['Parch'] + train['SibSp'] + 1

test['FSize'] = test['Parch'] + test['SibSp'] + 1
train['IsAlone'] = 1

train['IsAlone'].loc[train['FSize'] > 1] = 0



test['IsAlone'] = 1

test['IsAlone'].loc[test['FSize'] > 1] = 0
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train.head()
test.head()
train['Deck'] = le.fit_transform(train['Deck'])

train['Embarked'] = le.fit_transform(train['Embarked'])



test['Deck'] = le.fit_transform(test['Deck'])

test['Embarked'] = le.fit_transform(test['Embarked'])
train.isnull().sum()
test.isnull().sum()
train_col = train.columns

test_col = test.columns
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

imp_mean = IterativeImputer(random_state=0)
train = imp_mean.fit_transform(train)

test = imp_mean.fit_transform(test)
train = pd.DataFrame(train, columns=train_col)

test = pd.DataFrame(test, columns=test_col)
pclass_ohe_tr = pd.get_dummies(train['Pclass'], prefix='class', prefix_sep='_')

train.drop(['Pclass'], axis=1, inplace=True)

train = train.join(pclass_ohe_tr)



pclass_ohe_te = pd.get_dummies(test['Pclass'], prefix='class', prefix_sep='_')

test.drop(['Pclass'], axis=1, inplace=True)

test = test.join(pclass_ohe_te)
X = train.drop(['Survived'], axis=1)

Y = train['Survived']
X_col = X.columns

test_col = test.columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X.loc[:,['Age','Fare']] + test.loc[:,['Age','Fare']])

X.loc[:,['Age','Fare']] = scaler.transform(X.loc[:,['Age','Fare']])

test.loc[:,['Age','Fare']] = scaler.transform(test.loc[:,['Age','Fare']])
X = pd.DataFrame(X, columns=X_col)

test = pd.DataFrame(test, columns=test_col)
import seaborn as sns

sns.heatmap(train.corr())
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier(n_estimators=600, objective='binary:logistic',silent=True, nthread=1, colsample_bytree= 0.6,

 gamma= 2,

 max_depth= 4,

 min_child_weight= 1,

 subsample= 1.0)
xgb = XGBClassifier(

    objective= 'binary:logistic',

    nthread=4,

    seed=123

)
parameters = {

    'max_depth': range(2, 10, 1),

    'n_estimators': range(60, 220, 40),

    'learning_rate': [0.1, 0.01, 0.05, 0.5]

}
grid_search = GridSearchCV(

    estimator=xgb,

    param_grid=parameters,

    scoring = 'roc_auc',

    n_jobs = 10,

    cv = 10,

    verbose=True

)
grid_search.fit(X, Y)
grid_search.best_estimator_
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.05, max_delta_step=0, max_depth=6,

              min_child_weight=1, missing=nan, monotone_constraints='()',

              n_estimators=60, n_jobs=4, nthread=4, num_parallel_tree=1,

              random_state=123, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

              seed=123, subsample=1, tree_method='exact', validate_parameters=1,

              verbosity=None)
xgb.fit(X, Y)
pred = xgb.predict(test)
submission['Survived'] = pred
submission.to_csv('submission.csv')