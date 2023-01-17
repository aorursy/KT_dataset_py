import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.model_selection import KFold
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
plt.hist(train.loc[train['Survived'] == 0, 'Age'].dropna(), bins = 30, alpha = 0.5, label = '0')

plt.hist(train.loc[train['Survived'] == 1, 'Age'].dropna(), bins = 30, alpha = 0.5, label = '1')

plt.xlabel('Age')

plt.ylabel('count')

plt.legend(title = 'Survied')
sns.countplot(x = 'Pclass', hue = 'Survived', data = train)
sns.countplot(x = 'Sex', hue = 'Survived', data = train)
sns.countplot(x = 'SibSp', hue = 'Survived', data = train)
sns.countplot(x = 'Parch', hue = 'Survived', data = train)
sns.countplot(x = 'Embarked', hue = 'Survived', data = train)
data = pd.concat([train, test], sort=False)
data.isnull().sum()
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

train['FamilySize'] = data['FamilySize'][:len(train)]

test['FamilySize'] = data['FamilySize'][len(train):]

sns.countplot(x='FamilySize', data = train, hue = 'Survived')
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)
train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
clf = RandomForestClassifier(n_estimators=100, max_depth = 2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0, stratify = y_train)
categorical_features = ['Embarked', 'Pclass', 'Sex']
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



params = {

    'objective': 'binary'

}
model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval = 10, num_boost_round = 1000, early_stopping_rounds = 10)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
import optuna

from sklearn.metrics import log_loss



def objective(trial):

    params = {

        'objective':'binary', 

        'max_bin': trial.suggest_int('max_bin', 255, 500), 

        'learning_rate': 0.05, 

        'num_leaves': trial.suggest_int('num_leaves', 32, 128), 

    }

    

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

    

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference = lgb_train, categorical_feature=categorical_features)

    

    model = lgb.train(params, lgb_train,

                     valid_sets = [lgb_train, lgb_eval], 

                     verbose_eval = 10, 

                     num_boost_round = 1000, 

                     early_stopping_rounds = 10)

    

    y_pred_valid = model.predict(X_valid,

                                num_iteration=model.best_iteration)

    score = log_loss(y_valid, y_pred_valid)

    return score
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))

study.optimize(objective, n_trials = 40)
study.best_params
params = {

    'objective':'binary', 

    'max_bin': study.best_params['max_bin'], 

    'learning_rate': 0.05, 

    'num_leaves': study.best_params['num_leaves']

}



lgb_train = lgb.Dataset(X_train, y_train, 

                       categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference = lgb_train, 

                      categorical_feature=categorical_features)



model = lgb.train(params, lgb_train, valid_sets = [lgb_train, lgb_eval], 

                 verbose_eval = 10, 

                 num_boost_round = 1000, 

                 early_stopping_rounds = 10)



y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_preds = []

models = []

oof_train = np.zeros((len(X_train),))

cv = KFold(n_splits=5, shuffle=True, random_state=0)



categorical_features = ['Embarked', 'Pclass', 'Sex']



params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    lgb_train = lgb.Dataset(X_tr, y_tr,

                                             categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,

                                            categorical_feature=categorical_features)



    model = lgb.train(params, lgb_train,

                                   valid_sets=[lgb_train, lgb_eval],

                                   verbose_eval=10,

                                   num_boost_round=1000,

                                   early_stopping_rounds=10)





    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)



    y_preds.append(y_pred)

    models.append(model)
pd.DataFrame(oof_train).to_csv('oof_train_kfold.csv', index=False)



scores = [

    m.best_score['valid_1']['binary_logloss'] for m in models

]

score = sum(scores) / len(scores)

print('===CV scores===')

print(scores)

print(score)
from sklearn.metrics import accuracy_score





y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
y_pred = (y_pred > 0.5).astype(int)

sub['Survived'] = y_pred

sub.to_csv('submission_lightbgm.csv', index = False)