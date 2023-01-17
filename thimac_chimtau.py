import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

train_df = pd.read_csv("/kaggle/input/train.csv")

test_df = pd.read_csv("/kaggle/input/test.csv")
train_df["Survived_cat"] = train_df["Survived"].astype('category')

train_df.describe(include="all")
prediction = pd.DataFrame()

prediction['PassengerId'] = test_df['PassengerId'].copy()

prediction['Survived'] = 0

prediction.to_csv("submission.csv", index=False, sep=",", header=True)
from sklearn.utils import shuffle

from sklearn import preprocessing

from sklearn.utils import shuffle



def dataset(dataframe):

    try:

        Y = dataframe['Survived'].copy()

    except:

        Y = None

    X = dataframe[['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].copy()

    X['Age'] = X['Age'].fillna(value=29.7)

    X = X.fillna(method='ffill')

    return X,Y



train_df = pd.read_csv("/kaggle/input/train.csv")

test_df = pd.read_csv("/kaggle/input/test.csv")



train_df_it1 = pd.get_dummies(train_df.copy(), columns=["Pclass", "Embarked"])

test_df_it1 = pd.get_dummies(test_df.copy(), columns=["Pclass", "Embarked"])

train_df = shuffle(train_df_it1).reset_index(drop=True)

train_X, train_Y = dataset(train_df_it1[0:600])

val_X  , val_Y   = dataset(train_df_it1[600:])

test_X, _ = dataset(test_df_it1)



normalizer = preprocessing.StandardScaler().fit(train_X.append(val_X).append(test_X) )

train_X = normalizer.transform(train_X)

val_X = normalizer.transform(val_X)

test_X = normalizer.transform(test_X)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



clf = LogisticRegression(random_state=2, solver='lbfgs', max_iter=300, C=0.1).fit(train_X, train_Y)

val_pred = clf.predict(val_X) 

accuracy_score(val_Y, val_pred)
final_train_X, final_train_Y = dataset(train_df)

clf = LogisticRegression(random_state=1, solver='lbfgs', max_iter=1000, C=0.01).fit(final_train_X, final_train_Y)

pred = clf.predict(test_X)

prediction = pd.DataFrame()

prediction['PassengerId'] = test_df['PassengerId'].copy()

prediction['Survived'] = pred

prediction.to_csv("submission.csv", index=False, sep=",", header=True)
from sklearn.utils import shuffle

from sklearn import preprocessing

from sklearn.utils import shuffle



def fill_data(X):

    X['Age'] = X['Age'].fillna(value=29.7)

    X = X.fillna(method='ffill')

    return X



def dataset(dataframe, enc):

    try:

        Y = dataframe['Survived'].copy()

    except:

        Y = None

    

    X = dataframe[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()

    for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']:

        X[col] = enc[col].transform(X[col].copy())

    

    return X,Y



train_df = pd.read_csv("/kaggle/input/train.csv")

test_df = pd.read_csv("/kaggle/input/test.csv")



train_df_it2 = fill_data(train_df)

test_df_it2 = fill_data(test_df)



full_df = train_df_it2.append(test_df_it2)



enc = dict()

for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']: 

    enc_col = preprocessing.LabelEncoder()

    enc_col.fit(full_df[col].copy())

    enc[col] = enc_col



train_df_it2 = shuffle(train_df_it2).reset_index(drop=True)



train_X, train_Y = dataset(train_df_it2[0:600], enc)

val_X  , val_Y   = dataset(train_df_it2[600:], enc)

test_X, _ = dataset(test_df_it2, enc)
normalizer = preprocessing.StandardScaler().fit(train_X.append(val_X).append(test_X) )

train_X = normalizer.transform(train_X)

val_X = normalizer.transform(val_X)

test_X = normalizer.transform(test_X)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



clf = DecisionTreeClassifier()

clf = clf.fit(train_X, train_Y)

val_pred = clf.predict(val_X)

accuracy_score(val_Y, val_pred)
pred = clf.predict(test_X)

prediction = pd.DataFrame()

prediction['PassengerId'] = test_df['PassengerId'].copy()

prediction['Survived'] = pred

prediction.to_csv("submission.csv", index=False, sep=",", header=True)
from catboost import Pool, cv



def load_dataset(path):

    df = pd.read_csv(path)

    X = df[ ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] ].copy()

#     X['Age'] = X['Age'].fillna(29.7)

    X['Embarked'] = X['Embarked'].fillna('X')

    try:

        Y = df['Survived'].copy()

    except:

        Y =None

    return X, Y



X, Y = load_dataset("/kaggle/input/train.csv")

test_X, _ = load_dataset("/kaggle/input/test.csv")



cv_dataset = Pool(data=X,

                  label=Y,

                  cat_features=['Sex', 'Embarked', 'Pclass']

                  )

def objective(params):

    import random



    scores = cv(cv_dataset,

                params,

                seed=0,

                partition_random_seed=random.randint(1,10000),

                plot=False,

                fold_count=3)

    s = -scores['test-Accuracy-mean'].max()

    return s





from hyperopt import hp

from hyperopt import fmin, tpe, space_eval



params = {"iterations": 50,

        "learning_rate": hp.loguniform("learning_rate", -4, 0),

        "depth": hp.quniform("depth", 1, 5, 1),

        "loss_function": "Logloss",

        "verbose": False,

        'random_seed': 2,

        "eval_metric": 'Accuracy'}



best = fmin(fn=objective,

    space= params,

    algo=tpe.suggest,

    return_argmin=False,

    max_evals=100)



best
from catboost import CatBoostClassifier

best['iterations'] = 50

clf = CatBoostClassifier(**best)

clf.fit(cv_dataset)

df = pd.read_csv("/kaggle/input/test.csv")

test_X, _ = load_dataset("/kaggle/input/test.csv")

test_pool = Pool(data=test_X, cat_features=['Sex', 'Embarked', 'Pclass'])



pr = pd.DataFrame()

pr['PassengerId'] = df['PassengerId'].copy()

pr['Survived']  =  clf.predict(test_pool)

pr['Survived']  =   pr['Survived'].astype("int")

pr.to_csv("submission.csv", index=False, sep=",", header=True)
clf.get_feature_importance(prettified=True)