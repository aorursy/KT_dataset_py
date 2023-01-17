## Inintialize

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Data for training

train = pd.read_csv("../input/titanic/train.csv")

# Data for testign

test = pd.read_csv("../input/titanic/test.csv")

# Data to submit

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



train.head()
test.head()
gender_submission.head()
# The amount of data

print("train " + str(len(train)))

print("test " + str(len(test)))
# Connvert

data = pd.concat([train, test], sort=False)

data.head()
# Null

data.isnull().sum()
data_analyze = data.copy()

data_analyze.head()
data_analyze.describe()
import seaborn as sns



sns.countplot(x='Pclass', data=data_analyze, hue='Survived')



data_analyze['FamilySize'] = data_analyze['Parch'] + data_analyze['SibSp'] + 1

sns.countplot(x='FamilySize', data=data_analyze, hue='Survived')

# memo in_placeは副作用がある

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)



# 欠損値を愚直に埋める

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({ 'S': 0, 'C': 1, 'Q': 2 }).astype(int)



data['Fare'].fillna(np.mean(data['Fare']), inplace=True)



age_avg = data['Age'].mean()

age_std = data['Age'].std()

## TODO: これの意味理解したい

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)



## ぼっちかどうか

# 加工

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

# 特徴量生成

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



# Mr.等の加工

data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

data['Title'] = data['Title'].map(title_mapping)

data['Title'] = data['Title'].fillna(0)    



delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']

data.drop(delete_columns, axis=1, inplace=True)

data.head()
train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
y_train.head()
X_train.head()
X_test.head()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from sklearn.metrics import log_loss



def getCategoricalFeatures():

    return ['Embarked', 'Pclass', 'Sex']



def makeModel(X_train, y_train, X_valid, y_valid):

    categorical_features = getCategoricalFeatures()

    

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

    

    params = {

        'objective': 'binary',

        'max_bin': 427,

        'learning_rate': 0.05,

        'num_leaves': 79,

    }

    

    model = lgb.train(params, lgb_train,

                     valid_sets=[lgb_train, lgb_eval],

                     verbose_eval=10,

                     num_boost_round=1000,

                     early_stopping_rounds=10)

    

    return model

import optuna

from sklearn.model_selection import train_test_split



XX_train, XX_valid, yy_train, yy_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)



def objective(trial):

    params = {

        'objective': 'binary',

        'max_bin': trial.suggest_int('max_bin', 255, 500),

        'learning_rate': 0.05,

        'num_leaves': trial.suggest_int('num_leaves', 32, 128),

    }

    

    categorical_features = getCategoricalFeatures()

    

    lgb_train = lgb.Dataset(XX_train, yy_train, categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(XX_valid, yy_valid, reference=lgb_train, categorical_feature=categorical_features)

    

    model = lgb.train(params, lgb_train,

                     valid_sets=[lgb_train, lgb_eval],

                     verbose_eval=10,

                     num_boost_round=1000,

                     early_stopping_rounds=10)

        

    yy_pred_valid = model.predict(XX_valid, num_iteration=model.best_iteration)

    score = log_loss(yy_valid, yy_pred_valid)

    return score

    

def studyHyPara():

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    study.optimize(objective, n_trials=40)

    return study.best_params
# studyHyPara()
from sklearn.model_selection import KFold

from sklearn.metrics import log_loss, accuracy_score

def validation(X_train, y_train):

    y_preds = []

    models = []

    oof_train = np.zeros((len(X_train)))

    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    

    for train_index, valid_index in cv.split(X_train):

        X_tr = X_train.loc[train_index, :]

        X_val = X_train.loc[valid_index, :]

        y_tr = y_train[train_index]

        y_val = y_train[valid_index]

    

        model = makeModel(X_tr, y_tr, X_val, y_val)

        

        oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        

        y_preds.append(y_pred)

        models.append(model)

    

    scores = [

        m.best_score['valid_1']['binary_logloss'] for m in models

    ]

    score = sum(scores) / len(scores)

    print('===CV scores===')

    print(scores)

    print(score)    

    

    pd.DataFrame(oof_train).to_csv('oof_train_kfold.csv', index=False)

    

    y_pred_oof = (oof_train > 0.5).astype(int)

    print('===Score===')

    print(accuracy_score(y_train, y_pred_oof))

    

    # For submission

    y_sub = sum(y_preds) / len(y_preds)

    y_sub = (y_sub > 0.5).astype(int)

    return y_sub

        

y_pred = validation(X_train, y_train)
sub = gender_submission.copy()

sub['Survived'] = y_pred

y_pred[:10]

sub.to_csv("submission.csv", index=False)