import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np



# データの読み込み

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



# まとめて処理する

data = pd.concat([train, test], sort=True)



data['Sex'].replace(['male','female'],[0, 1], inplace=True) #文字列の置換

data['Embarked'].fillna(('S'), inplace=True) #欠損値を埋める

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) #文字列の置換

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(100, inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1 #新しい特徴量の作成

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Age'] = data['Age'].astype(int) #floatじゃデータが多すぎるのでintにして少しだけ情報量を落とした

data['Fare'] = data['Fare'].astype(int)

delete_columns = ['Cabin','Ticket','PassengerId'] #明らかに使えないので消した

data.drop(delete_columns, axis = 1, inplace = True)

data.head()
data = data.reset_index()

data['Middle_Name'] = 0

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
import seaborn as sns

sns.countplot(x='Middle_Name',hue='Survived',data=data)
data['Last_Name'] = "hoge"

name_set = set()

for i in range(len(data['Name'])):

    l = data['Name'][i].split()

    for j in range(len(l)):

        if '.' in l[j]:

            data['Last_Name'][i] = l[j+1]

            name_set.add(l[j+1])

data.head()
len(name_set)
plt.figure(figsize=(30,10))

sns.countplot(x='Last_Name',hue='Survived',data=data,)
import collections

data['Name_count'] = 0

c = collections.Counter(data['Last_Name'])

for i in range(len(data['Last_Name'])):

    data['Name_count'][i] = c[data['Last_Name'][i]]

data.head()
plt.figure(figsize=(30,10))

sns.countplot(x='Name_count',hue='Survived',data=data,)
delete_columns = ['index','Name','Last_Name']

data.drop(delete_columns, axis = 1, inplace = True)

data.head()
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





cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) #コレだけでデータを分割できる

for fold_id, (train_index, valid_index) in enumerate(cv.split(x_train, y_train)):

    x_tr = x_train.loc[train_index, :]

    x_val = x_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    print(f'fold: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')
import optuna

import lightgbm as lgb

from sklearn.metrics import accuracy_score

categorical_features = ['Pclass','Age','Middle_Name']

def objective(trial):

    models = [] #学習したモデルを記録

    oof_train = np.zeros((len(x_train),)) #学習精度の算出に利用する

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 512),

        'learning_rate': 0.05,

        'num_leaves': trial.suggest_int('num_leaves', 64,256),

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



        y_pred = model.predict(x_val, num_iteration=model.best_iteration) #validationデータの予測値で精度を評価する

        oof_train[valid_index] = y_pred

        models.append(model) #この分割で学習したモデルを記録

    y_pred_oof = (oof_train>0.5).astype(int) #0,1に置き換える

    acc = accuracy_score(y_train, y_pred_oof) #誤差関数　今回は誤り率とした

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

    'learning_rate':   0.05,

    'num_leaves': 111,

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
y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

y_sub[:10]
sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub['Survived'] = y_sub

sub.to_csv("submission_LightGBM.csv", index = False)

sub.head()
delete_columns = ['IsAlone','SibSp','Sex','Parch']

data.drop(delete_columns, axis = 1, inplace = True)

data.head()
train = train = data[:len(train)]

test = data[len(train):]

train = train.reset_index()

test = test.reset_index()

train.drop('index',axis=1,inplace=True)

test.drop('index',axis=1,inplace=True)

y_train = train['Survived']

x_train = train.drop('Survived', axis = 1)

x_test = test.drop('Survived', axis = 1)
import optuna

from sklearn.metrics import accuracy_score

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

categorical_feature = ['Embarked','Fare','Middle_Name']

def objective(trial):

    models = []

    oof_train = np.zeros((len(x_train),))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 4, 32), #ここを変更していく 大きいと精度が上がりやすいが、過学習をなくすには小さくする

        'learning_rate': trial.suggest_loguniform("learning_rate",0.005,0.02), #ここも調整する

        'num_leaves': trial.suggest_int('num_leaves', 4, 16), #ここを変更していく 大きいと精度が上がりやすいが、過学習をなくすには小さくする

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

            early_stopping_rounds=20

        )



        y_pred = model.predict(x_val, num_iteration=model.best_iteration)

        oof_train[valid_index] = y_pred

        models.append(model)

    y_pred_oof = (oof_train>0.5).astype(int)

    acc = accuracy_score(y_train, y_pred_oof)

    return 1.0 - acc
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

study.optimize(objective, n_trials = 20)
study.best_params
y_preds = []

x_preds = []

models = []

oof_train = np.zeros((len(x_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

params = {

    'objective': 'binary',

    'max_bin': 13,

    'learning_rate': 0.017232299486550178,

    'num_leaves': 4,

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

y_sub[:10]
sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub['Survived'] = y_sub

sub.to_csv("submission_LightGBM_tuning.csv", index = False)

sub.head()