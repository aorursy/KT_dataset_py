import pandas as pd

import numpy as np

from catboost import CatBoostClassifier

from sklearn.preprocessing import OneHotEncoder



def split_name(n):

    fn = n.split(", ")

    a = fn[0]

    fn = fn[1]

    title = fn.split(" ")[0]

    fn = " ".join(fn.split(" ")[1:])

    return title, fn, a



def parse_name(name):

    name=name.replace('"', '')

    lst = name.split("(")

    fn = split_name(lst[0])

    if len(lst) == 1:

        return fn[0], fn[1], fn[2], ""

    else:

        real_name = lst[1][:-1]

        return fn[0], fn[1], fn[2], real_name

    

def gen_data(df):

    X = df[ ['Age', 'Name', 'Pclass', 'Fare', 'Sex']].copy()    

    X['Age'] = X['Age'].fillna(100)

    X['Age'] = np.log(X['Age'] + 0.1)

    X.Fare = X['Fare'].fillna(1e-1 - 1.0)

    X['Fare'] = np.log(X['Fare'] + 1.0).astype("float")    

    X['Family_Size']  = df.SibSp + df.Parch

    ln1 = []

    ln2 = []

    for i in range(X.shape[0]):

        n1, _, n2, _ = parse_name(X['Name'][i])

        ln1.append(n1)

        ln2.append(n2)

    X['Name_1'] = ln1

    X['Name_2'] = ln2

    X = X.drop(columns=["Name"])

    

    try:

        Y = df['Survived'].copy()

    except:

        Y =None

    return X, Y



train_df = pd.read_csv("/kaggle/input/train.csv")

test_df = pd.read_csv("/kaggle/input/test.csv")

num_features = ['Fare']

cat_features = ['Pclass', 'Name_1', 'Name_2']

features = num_features + cat_features



train_XX, train_yy = gen_data(train_df)

all_features = train_XX.columns

test_X, _ = gen_data(test_df)

train_XX = train_XX[features]

test_X = test_X[features]

all_X = pd.concat([train_XX, test_X])
from catboost import Pool, cv



cv_dataset = Pool(data=train_XX,

                  label=train_yy,

                  feature_names = list(features),

                  cat_features=cat_features

                 )



params = {"iterations": 1000,

          "depth": 2,

          "learning_rate": 0.01,

          "eval_metric": "Accuracy",

          "loss_function": "Logloss",

          "random_seed": 1,

          "verbose": False}



scores = cv(cv_dataset,

            params,

            fold_count=5, 

            plot="True",

            stratified=False

           )
params['iterations'] = 1000

params['random_seed'] = 42

clf = CatBoostClassifier(**params)

clf.fit(train_XX, train_yy, cat_features=cat_features)

test_y = clf.predict_proba(test_X)



pr = pd.DataFrame()

df = pd.read_csv("/kaggle/input/test.csv")

pr['PassengerId'] = df['PassengerId'].copy()

pr['Survived']  =   test_y.argmax(axis=-1)

pr.to_csv("submission.csv", index=False, sep=",", header=True)
clf.get_feature_importance(prettified=True)