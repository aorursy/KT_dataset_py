# 特徴量の準備



%matplotlib inline

import numpy as np

import pandas as pd





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket']

data.drop(delete_columns, axis = 1, inplace = True)
data.head()
data.isnull().sum()
data = data.reset_index()
data['Middle_Name'] = -1
data.head()
data['Name'][0]
for i in range(len(data['Name'])):

    val = 0

    if 'Miss' in data['Name'][i]:

        val = 1

    elif 'Master' in data['Name'][i]:

        val = 2

    elif 'Mrs' in data['Name'][i]:

        val = 3

    elif 'Mr' in data['Name'][i]:

        val = 4

    data['Middle_Name'][i] = val
data.head()
data.drop(['Name','index'], axis = 1, inplace = True)
data.head()
train = train = data[:len(train)]

test = data[len(train):]
train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

test.head()
y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
from sklearn.model_selection import StratifiedKFold





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')

categorical_features = ['Embarked', 'Pclass', 'Middle_Name']
import lightgbm as lgb
import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 500),

        'learning_rate': 0.05,

        'num_leaves': trial.suggest_int('num_leaves', 32, 128),

    }

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        lgb_train = lgb.Dataset(x_tr, y_tr, categorical_feature=categorical_features)

        lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train, categorical_feature=categorical_features)



        model = lgb.train(

            params, lgb_train,

            valid_sets=[lgb_train, lgb_eval],

            verbose_eval=10,

            num_boost_round=10000,

            early_stopping_rounds=10

        )



        y_pred = model.predict(x_val, num_iteration=model.best_iteration)

        oof_train[valid_index] = y_pred

        models.append(model)

    y_pred_oof = (oof_train>0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=20)
study.best_params
y_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

params = {

    'objective': 'binary',

    'max_bin': 427,

    'learning_rate': 0.05,

    'num_leaves': 79,

}

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    lgb_train = lgb.Dataset(x_tr, y_tr, categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train, categorical_feature=categorical_features)



    model = lgb.train(

        params, lgb_train,

        valid_sets=[lgb_train, lgb_eval],

        verbose_eval=10,

        num_boost_round=10000,

        early_stopping_rounds=10

    )



    y_pred = model.predict(x_val, num_iteration=model.best_iteration)

    oof_train[valid_index] = y_pred

    y_preds.append(y_pred)

    models.append(model)
scores = [

    m.best_score['valid_1']['binary_logloss'] for m in models

]

score = sum(scores) / len(scores)

print('===CV scores===')

print(scores)

print(score)
feature_importance = [0 for i in range(len(x_train.columns))]

for m in models:

    for i in range(len(m.feature_importance())):

        feature_importance[i]+=m.feature_importance()[i]

feature_importance
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train>0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
# 特徴量の準備



%matplotlib inline

import numpy as np

import pandas as pd





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket']

data.drop(delete_columns, axis = 1, inplace = True)
data = data.reset_index()

data['Middle_Name'] = -1

for i in range(len(data['Name'])):

    val = 0

    if 'Miss' in data['Name'][i]:

        val = 1

    elif 'Master' in data['Name'][i]:

        val = 2

    elif 'Mrs' in data['Name'][i]:

        val = 3

    elif 'Mr' in data['Name'][i]:

        val = 4

    data['Middle_Name'][i] = val

data.head(10)
data.drop(['Name','index','PassengerId','IsAlone','SibSp','Sex','Parch'], axis = 1, inplace = True)
train = train = data[:len(train)]

test = data[len(train):]

train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
from sklearn.model_selection import StratifiedKFold





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')

categorical_features = ['Embarked', 'Pclass', 'Middle_Name']
import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 500),

        'learning_rate': 0.05,

        'num_leaves': trial.suggest_int('num_leaves', 32, 128),

    }

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        lgb_train = lgb.Dataset(x_tr, y_tr, categorical_feature=categorical_features)

        lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train, categorical_feature=categorical_features)



        model = lgb.train(

            params, lgb_train,

            valid_sets=[lgb_train, lgb_eval],

            verbose_eval=10,

            num_boost_round=10000,

            early_stopping_rounds=10

        )



        y_pred = model.predict(x_val, num_iteration=model.best_iteration)

        oof_train[valid_index] = y_pred

        models.append(model)

    y_pred_oof = (oof_train>0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials=20)
study.best_params
y_preds = []

x_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

params = {

    'objective': 'binary',

    'max_bin': 427,

    'learning_rate': 0.05,

    'num_leaves': 79,

}

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    lgb_train = lgb.Dataset(x_tr, y_tr, categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train, categorical_feature=categorical_features)



    model = lgb.train(

        params, lgb_train,

        valid_sets=[lgb_train, lgb_eval],

        verbose_eval=10,

        num_boost_round=10000,

        early_stopping_rounds=10

    )



    y_pred = model.predict(x_test, num_iteration=model.best_iteration)

    oof_train[valid_index] = model.predict(x_val, num_iteration=model.best_iteration)

    y_preds.append(y_pred)

    x_preds.append(model.predict(x_train,num_iteration=model.best_iteration))

    models.append(model)
scores = [

    m.best_score['valid_1']['binary_logloss'] for m in models

]

score = sum(scores) / len(scores)

print('===CV scores===')

print(scores)

print(score)
feature_importance = [0 for i in range(len(x_train.columns))]

for m in models:

    for i in range(len(m.feature_importance())):

        feature_importance[i]+=m.feature_importance()[i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train>0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub['Survived'] = y_sub

sub.to_csv("submission_lightgbm.csv", index = False)

sub.head()
lgb_pred_x = sum(x_preds) / len(x_preds)

print(lgb_pred_x[:10])

lgb_pred_y = sum(y_preds) / len(y_preds)

print(lgb_pred_y[:10])
from sklearn.ensemble import RandomForestClassifier
import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    min_samples_split = trial.suggest_int("min_samples_split", 8, 32)

    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 256, 4))

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = min_samples_split, 

                                max_leaf_nodes = max_leaf_nodes,

                                criterion = criterion,

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        y_pred = model.predict(x_val)

        oof_train[valid_index] = y_pred

        models.append(model)

    y_pred_oof = (oof_train > 0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials = 50)
study.best_params
y_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = 23, 

                                max_leaf_nodes = 224,

                                criterion = 'gini',

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        y_pred = model.predict(x_val)

        oof_train[valid_index] = y_pred

        y_preds.append(model.predict(x_test))

        models.append(model)
feature_importance = [0 for i in range(len(x_train.columns))]



for m in models:

    for i in range(len(m.feature_importances_)):

        feature_importance[i]+=m.feature_importances_[i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train>0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

y_sub[:10]
sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub['Survived'] = y_sub

sub.to_csv("submission_RandomForest.csv", index = False)

sub.head()
# 特徴量の準備



%matplotlib inline

import numpy as np

import pandas as pd





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket']

data.drop(delete_columns, axis = 1, inplace = True)
data = data.reset_index()

data['Middle_Name'] = -1

for i in range(len(data['Name'])):

    val = 0

    if 'Miss' in data['Name'][i]:

        val = 1

    elif 'Master' in data['Name'][i]:

        val = 2

    elif 'Mrs' in data['Name'][i]:

        val = 3

    elif 'Mr' in data['Name'][i]:

        val = 4

    data['Middle_Name'][i] = val

data.head(10)
data.drop(['Name','index','PassengerId','IsAlone','SibSp','Parch'], axis = 1, inplace = True)
train = train = data[:len(train)]

test = data[len(train):]

train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
from sklearn.model_selection import StratifiedKFold





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')

import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    y_preds = []

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    min_samples_split = trial.suggest_int("min_samples_split", 8, 32)

    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 256, 4))

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = min_samples_split, 

                                max_leaf_nodes = max_leaf_nodes,

                                criterion = criterion,

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        y_pred = model.predict(x_val)

        oof_train[valid_index] = y_pred

        y_preds.append(model.predict(x_train))

        models.append(model)

    y_pred_oof = (oof_train > 0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials = 50)
study.best_params
y_preds = []

x_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = 8, 

                                max_leaf_nodes = 152,

                                criterion = 'entropy',

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        oof_train[valid_index] = model.predict(x_val)

        y_preds.append(model.predict(x_test))

        x_preds.append(model.predict(x_train))

        models.append(model)
feature_importance = [0 for i in range(len(x_train.columns))]



for m in models:

    for i in range(len(m.feature_importances_)):

        feature_importance[i]+=m.feature_importances_[i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
y_preds[0]
y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

y_sub[:10]
sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub['Survived'] = y_sub

sub.to_csv("submission_RandomForest_skfold.csv", index = False)

sub.head(20)
RFclf_pred_x = sum(x_preds) / len(x_preds)

print(RFclf_pred_x[:10])

RFclf_pred_y = sum(y_preds) / len(y_preds)

print(RFclf_pred_y[:10])
from sklearn.linear_model import LogisticRegression
# 特徴量の準備



%matplotlib inline

import numpy as np

import pandas as pd





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket']

data.drop(delete_columns, axis = 1, inplace = True)
data = data.reset_index()

data['Middle_Name'] = -1

for i in range(len(data['Name'])):

    val = 0

    if 'Miss' in data['Name'][i]:

        val = 1

    elif 'Master' in data['Name'][i]:

        val = 2

    elif 'Mrs' in data['Name'][i]:

        val = 3

    elif 'Mr' in data['Name'][i]:

        val = 4

    data['Middle_Name'][i] = val

data.head(10)
data.drop(['Name','index','PassengerId'], axis = 1, inplace = True)
train = train = data[:len(train)]

test = data[len(train):]

train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
from sklearn.model_selection import StratifiedKFold





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')

import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    C = trial.suggest_loguniform("C",0.00001,10)

    max_iter = trial.suggest_int("max_iter",100,10000)

    

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = LogisticRegression(random_state=0,penalty='l2',C=C,solver='lbfgs',max_iter=max_iter)

        model.fit(x_tr,y_tr)

        y_pred = model.predict(x_val)

        oof_train[valid_index] = y_pred

        models.append(model)

    y_pred_oof = (oof_train > 0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials = 50)
study.best_params
y_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = LogisticRegression(random_state=0,penalty='l2',C= 0.5627949975223141,solver='lbfgs',max_iter=2322)

        model.fit(x_tr,y_tr)

        oof_train[valid_index] = model.predict(x_val)

        y_preds.append(model.predict(x_test))

        models.append(model)
feature_importance = [0 for i in range(len(x_train.columns))]



for m in models:

    print(m.coef_[0])

    for i in range(len(m.coef_[0])):

        feature_importance[i]+=m.coef_[0][i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
# 特徴量の準備



%matplotlib inline

import numpy as np

import pandas as pd





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket']

data.drop(delete_columns, axis = 1, inplace = True)
data = data.reset_index()

data['Middle_Name'] = -1

for i in range(len(data['Name'])):

    val = 0

    if 'Miss' in data['Name'][i]:

        val = 1

    elif 'Master' in data['Name'][i]:

        val = 2

    elif 'Mrs' in data['Name'][i]:

        val = 3

    elif 'Mr' in data['Name'][i]:

        val = 4

    data['Middle_Name'][i] = val

data.head(10)
data.drop(['Name','index','PassengerId','Age','Fare','Parch'], axis = 1, inplace = True)
train = train = data[:len(train)]

test = data[len(train):]

train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
from sklearn.model_selection import StratifiedKFold





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')

import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    C = trial.suggest_loguniform("C",0.00001,10)

    max_iter = trial.suggest_int("max_iter",100,10000)

    

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = LogisticRegression(random_state=0,penalty='l2',C=C,solver='lbfgs',max_iter=max_iter)

        model.fit(x_tr,y_tr)

        y_pred = model.predict(x_val)

        oof_train[valid_index] = y_pred

        models.append(model)

    y_pred_oof = (oof_train > 0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials = 50)
study.best_params
y_preds = []

x_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = LogisticRegression(random_state=0,penalty='l2',C= 1.0386580256500273,solver='lbfgs',max_iter=6316)

        model.fit(x_tr,y_tr)

        oof_train[valid_index] = model.predict(x_val)

        y_preds.append(model.predict(x_test))

        x_preds.append(model.predict(x_train))

        models.append(model)
feature_importance = [0 for i in range(len(x_train.columns))]



for m in models:

    print(m.coef_[0])

    for i in range(len(m.coef_[0])):

        feature_importance[i]+=m.coef_[0][i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
LRclf_pred_x = sum(x_preds) / len(x_preds)

print(LRclf_pred_x[:10])

LRclf_pred_y = sum(y_preds) / len(y_preds)

print(LRclf_pred_y[:10])
# 特徴量の準備



%matplotlib inline

import numpy as np

import pandas as pd





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket']

data.drop(delete_columns, axis = 1, inplace = True)
data = data.reset_index()

data['Middle_Name'] = -1

for i in range(len(data['Name'])):

    val = 0

    if 'Miss' in data['Name'][i]:

        val = 1

    elif 'Master' in data['Name'][i]:

        val = 2

    elif 'Mrs' in data['Name'][i]:

        val = 3

    elif 'Mr' in data['Name'][i]:

        val = 4

    data['Middle_Name'][i] = val

data.head(10)
data.drop(['Name','index','PassengerId'], axis = 1, inplace = True)
train = train = data[:len(train)]

test = data[len(train):]

train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
x_train['lgb'] = lgb_pred_x

x_train['rfc'] = RFclf_pred_x

x_train['lr'] = LRclf_pred_x

x_test['lgb'] = lgb_pred_y

x_test['rfc'] = RFclf_pred_y

x_test['lr'] = LRclf_pred_y
x_train.head()
x_test.head()
from sklearn.model_selection import StratifiedKFold





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')
import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    y_preds = []

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    min_samples_split = trial.suggest_int("min_samples_split", 8, 32)

    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 256, 4))

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = min_samples_split, 

                                max_leaf_nodes = max_leaf_nodes,

                                criterion = criterion,

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        y_pred = model.predict(x_val)

        oof_train[valid_index] = y_pred

        y_preds.append(model.predict(x_train))

        models.append(model)

    y_pred_oof = (oof_train > 0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials = 50)
study.best_params
y_preds = []

x_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = 32, 

                                max_leaf_nodes = 72,

                                criterion = 'entropy',

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        oof_train[valid_index] = model.predict(x_val)

        y_preds.append(model.predict(x_test))

        x_preds.append(model.predict(x_train))

        models.append(model)
feature_importance = [0 for i in range(len(x_train.columns))]



for m in models:

    for i in range(len(m.feature_importances_)):

        feature_importance[i]+=m.feature_importances_[i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

y_sub[:10]

sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub['Survived'] = y_sub

sub.to_csv("submission_ensemble_modified.csv", index = False)

sub.head(20)
# 特徴量の準備



%matplotlib inline

import numpy as np

import pandas as pd





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int)

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket']

data.drop(delete_columns, axis = 1, inplace = True)
data = data.reset_index()

data['Middle_Name'] = -1

for i in range(len(data['Name'])):

    val = 0

    if 'Miss' in data['Name'][i]:

        val = 1

    elif 'Master' in data['Name'][i]:

        val = 2

    elif 'Mrs' in data['Name'][i]:

        val = 3

    elif 'Mr' in data['Name'][i]:

        val = 4

    data['Middle_Name'][i] = val

data.head(10)
data.drop(['Name','index','PassengerId','Embarked','Parch','SibSp','IsAlone','FamilySize'], axis = 1, inplace = True)
train = train = data[:len(train)]

test = data[len(train):]

train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
x_train['lgb'] = lgb_pred_x

x_train['rfc'] = RFclf_pred_x

x_train['lr'] = LRclf_pred_x

x_test['lgb'] = lgb_pred_y

x_test['rfc'] = RFclf_pred_y

x_test['lr'] = LRclf_pred_y
from sklearn.model_selection import StratifiedKFold





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')
import optuna

from sklearn.metrics import accuracy_score

def objective(trial):

    y_preds = []

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    min_samples_split = trial.suggest_int("min_samples_split", 8, 32)

    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 256, 4))

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    

    for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = min_samples_split, 

                                max_leaf_nodes = max_leaf_nodes,

                                criterion = criterion,

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        y_pred = model.predict(x_val)

        oof_train[valid_index] = y_pred

        y_preds.append(model.predict(x_train))

        models.append(model)

    y_pred_oof = (oof_train > 0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials = 200)
study.best_params
y_preds = []

x_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

        x_tr = x_train.loc[train_index, :]

        x_val = x_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]



        model = RandomForestClassifier(min_samples_split = 21, 

                                max_leaf_nodes = 12,

                                criterion = 'gini',

                                n_estimators=100,

                                random_state=0)

        model.fit(x_tr,y_tr)

        oof_train[valid_index] = model.predict(x_val)

        y_preds.append(model.predict(x_test))

        x_preds.append(model.predict(x_train))

        models.append(model)
feature_importance = [0 for i in range(len(x_train.columns))]



for m in models:

    for i in range(len(m.feature_importances_)):

        feature_importance[i]+=m.feature_importances_[i]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(111)

ax.bar(range(len(x_train.columns)),feature_importance,tick_label=x_train.columns)
from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

y_sub[:10]

sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub['Survived'] = y_sub

sub.to_csv("submission_last.csv", index = False)

sub.head(20)