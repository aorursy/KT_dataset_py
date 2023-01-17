import pandas as pd

import pandas_profiling

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.profile_report()
sns.countplot(x='SibSp', hue='Survived', data=train)

plt.legend(loc='upper right', title='Survived')
sns.countplot(x='Parch', hue='Survived', data=train)

plt.legend(loc='upper right', title='Survived')
data = pd.concat([train, test], sort=False)
pclass_group = data.groupby('Pclass')

#pclass_group.mean()['Fare'][1]

#pclass_group.median()['Fare']



mean = pclass_group.median()['Fare']



m1 = (data['Pclass'] == 1)

data.loc[m1,'Fare'] = data.loc[m1,'Fare'].fillna(mean[1])



m2 = (data['Pclass'] == 2)

data.loc[m2,'Fare'] = data.loc[m2,'Fare'].fillna(mean[2])



m3 = (data['Pclass'] == 3)

data.loc[m3,'Fare'] = data.loc[m3,'Fare'].fillna(mean[3])





#data.iloc[:,10]=np.nan

#df[df['A'].isnull()]
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

#data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
data.head()
delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

y_test = test['Survived']

X_test = test.drop('Survived', axis=1)

X_train.head()
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission_randomforest.csv', index=False)

sub.head()
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
categorical_features=['Embarked', 'Pclass', 'Sex']
import lightgbm as lgb

import optuna

from sklearn.metrics import log_loss



def objective(trial):

  params = {

      'objective': 'binary',

      'max_bin': trial.suggest_int('max_bin', 255, 500),

      'learning_rate': 0.05,

      'num_leaves': trial.suggest_int('num_leaves', 32, 128)

  }



  lgb_train =lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

  lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



  model = lgb.train(params, lgb_train, 

                    valid_sets=[lgb_train, lgb_eval],

                    verbose_eval=10,

                    num_boost_round=1000,

                    early_stopping_rounds=10)

  

  y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

  score = log_loss(y_valid, y_pred_valid)

  return score



study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=40)

study.best_params
params = {

    'objective': 'binary',

    'max_bin': study.best_params['max_bin'],

    'learning_rate': 0.05,

    'num_leaves': study.best_params['num_leaves']

}



lgb_train = lgb.Dataset(X_train, y_train,

                                         categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train,

                                         categorical_feature=categorical_features)



model = lgb.train(params, lgb_train,

                               valid_sets=[lgb_train, lgb_eval],

                               verbose_eval=10,

                               num_boost_round=1000,

                               early_stopping_rounds=10)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.5).astype(int)



sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = y_pred

sub.to_csv('submission_lightgbm_optuna.csv', index=False)



sub
from catboost import CatBoost

from catboost import Pool



train_pool = Pool(X_train, label=y_train)

test_pool = Pool(X_test, label=y_test)



params = {

    'loss_function': 'Logloss',

    'num_boost_round': 100

}



model = CatBoost(params)

model.fit(train_pool)

y_pred = model.predict(test_pool, prediction_type='Class')
y_pred = (y_pred > 0.5).astype(int)



sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = y_pred

sub.to_csv('submission_catboost.csv', index=False)



sub
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

X_train_ss = ss.fit_transform(X_train)

X_test_ss = ss.fit_transform(X_test)



model = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')

model.fit(X_train_ss, y_train)



y_pred = model.predict(X_test_ss)

y_pred

y_pred = (y_pred > 0.5).astype(int)

y_pred 
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = y_pred

sub.to_csv('submission_knn.csv', index=False)



sub
randomforest = pd.read_csv('/kaggle/working/submission_randomforest.csv')

lightgbm = pd.read_csv('/kaggle/working/submission_lightgbm_optuna.csv')

catboost = pd.read_csv('/kaggle/working/submission_catboost.csv')

knn = pd.read_csv('/kaggle/working/submission_knn.csv')

df = pd.DataFrame({'sub_lgb': lightgbm['Survived'].values,

                   'sub_rf': randomforest['Survived'].values,

                   'sub_knn': knn['Survived'].values,

                   'sub_cb': catboost['Survived'].values})

df.head()
df.corr()
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = lightgbm['Survived'] + randomforest['Survived'] + catboost['Survived'] + knn['Survived']

sub.head()
sub['Survived'] = (sub['Survived'] >= 3).astype(int)

sub.to_csv('submission_lightgbm_ensemble.csv', index=False)

sub