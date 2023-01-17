# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
def missed_table(df): 

    null_val = df.isnull().sum()

    percent = 100 * df.isnull().sum() / len(df)

    missed_table = pd.concat([null_val, percent], axis=1)

    missed_table_ren_columns = missed_table.rename(

    columns = {0 : 'missed counts', 1 : '%'})

    return missed_table_ren_columns
missed_table(train)
missed_table(test)
import matplotlib.pyplot as plt
df = train.Cabin.fillna('Z').copy()

df = pd.DataFrame([x[0] for x  in df])

train['Cabin2'] = df
train.groupby('Cabin2').count()
train.groupby('Cabin2').mean()
plt.plot(train.Fare, train.Cabin2, 'o', alpha=0.2)
train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")

train["Cabin"] = train["Cabin"].fillna("C85")



test["Age"] = test["Age"].fillna(test["Age"].median())

test["Embarked"] = test["Embarked"].fillna("S")

test["Fare"] = test["Fare"].fillna("7.9250")
train.head()
# categorical列のみを持ってくる

cat_train_cols = list(set(train.columns) - set(train._get_numeric_data().columns))

cat_test_cols = list(set(test.columns) - set(test._get_numeric_data().columns))
cat_train_cols
for col in cat_train_cols:

    labels, uniques = pd.factorize(train[col])

    train[col] = labels
for col in cat_test_cols:

    labels, uniques = pd.factorize(test[col])

    test[col] = labels
train.head()
test.head()
from sklearn import tree
# 「train」の目的変数と説明変数の値を取得

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

 

# 決定木の作成

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

 

# 「test」の説明変数の値を取得

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

 

# 「test」の説明変数を使って「my_tree_one」のモデルで予測

my_prediction = my_tree_one.predict(test_features)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し

my_solution.to_csv("dt_1.csv", index_label = ["PassengerId"])
# 追加となった項目も含めて予測モデルその2で使う値を取り出す

feature_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]

features_two = train[feature_col].values



# 決定木の作成

max_depth = 10

min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(

    max_depth = max_depth,

    min_samples_split = min_samples_split,

    random_state = 1

)

my_tree_two = my_tree_two.fit(features_two, target)
# tsetから「その2」で使う項目の値を取り出す

test_features_2 = test[feature_col].values

 

# 「その2」の決定木を使って予測をしてCSVへ書き出す

my_prediction_tree_two = my_tree_two.predict(test_features_2)

PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns = ["Survived"])

my_solution_tree_two.to_csv("dt_2.csv", index_label = ["PassengerId"])