%matplotlib inline

import numpy as np

import pandas as pd



np.random.seed(0)
!ls ../input
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

gender_submission = pd.read_csv("../input/gender_submission.csv")
gender_submission.head()
train.head()
test.head()
data = pd.concat([train, test], sort=True)
data.head()
print(len(train), len(test), len(data))
data.isnull().sum()
data['Pclass'].value_counts()
data['Sex'].replace(['male','female'],[0, 1], inplace=True)
data['Embarked'].value_counts()
data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['Family_Size'] = data['Parch'] + data['SibSp'] + 1

train['Family_Size'] = data['Family_Size'][:len(train)]

test['Family_Size'] = data['Family_Size'][len(train):]



import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x='Family_Size', data = train, hue = 'Survived')
data['IsAlone'] = 0

data.loc[data['Family_Size'] == 1, 'IsAlone'] = 1
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis = 1, inplace = True)
train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis = 1)

X_test = test.drop('Survived', axis = 1)
X_train
y_train
# Stats

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

# Data processing, metrics and modeling

from sklearn.model_selection import train_test_split, RandomizedSearchCV

import lightgbm as lgbm
fit_params = {"early_stopping_rounds" : 100, 

             "eval_metric" : 'auc', 

             "eval_set" : [(X_train,y_train)],

             'eval_names': ['valid'],

             'verbose': 0,

             'categorical_feature': 'auto'}



param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],

              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],

              'num_leaves': sp_randint(6, 50), 

              'min_child_samples': sp_randint(100, 500), 

              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

              'subsample': sp_uniform(loc = 0.2, scale = 0.8), 

              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],

              'colsample_bytree': sp_uniform(loc = 0.4, scale = 0.6),

              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}



n_iter = 500
lgbm_clf = lgbm.LGBMClassifier(random_state = 42, silent = True, metric = 'None', n_jobs = 4)

grid_search = RandomizedSearchCV(

    estimator = lgbm_clf, param_distributions = param_test, 

    n_iter = n_iter,

    scoring = 'accuracy',

    cv = 5,

    refit = True,

    random_state = 42,

    verbose = True)



grid_search.fit(X_train, y_train, **fit_params)

opt_parameters = grid_search.best_params_
opt_parameters
clf = lgbm.LGBMClassifier(**opt_parameters).fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred[:20]
sub = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv("submission.csv", index = False)

!ls .