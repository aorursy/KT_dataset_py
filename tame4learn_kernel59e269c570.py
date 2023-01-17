# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# データの読み込み

gender_submission_path = "/kaggle/input/titanic/gender_submission.csv"

test_path = "/kaggle/input/titanic/test.csv"

train_path = "/kaggle/input/titanic/train.csv"



gender_submission = pd.read_csv(gender_submission_path)

train = pd.read_csv(train_path)

test = pd.read_csv(test_path)





# Any results you write to the current directory are saved as output.
def kesson_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum()/len(df)

        kesson_table = pd.concat([null_val, percent], axis=1)

        kesson_table_ren_columns = kesson_table.rename(

        columns = {0 : '欠損数', 1 : '%'})

#         df.info()

        return kesson_table_ren_columns

 

# kesson_table(train)

# kesson_table(test)

train.describe()

# train["Parch"].unique()
'''データの前処理'''

# 欠損値を埋める Cabinは使わないので、AgeとEnbarkedを対応する

# Ageは中央値を使う

train["Age"] = train["Age"].fillna(train["Age"].median())

# Embarkedは、一番多いSにします。

train["Embarked"] = train["Embarked"].fillna("S")

test["Age"] = train["Age"].fillna(train["Age"].median())

test["Embarked"] = train["Embarked"].fillna("S")

test["Fare"] = train["Fare"].fillna(train["Fare"].median())



# SexとEmbarkedのOne-Hotエンコーディング 数値に変換してもいいし、それぞれでone-hot表現にしてもいい

train = pd.get_dummies(train, columns=['Sex', 'Embarked'])

test = pd.get_dummies(test, columns=['Sex', 'Embarked'])



# 不要な列の削除

# train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

# test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)



# trainの表示

display(test)

test.isnull().sum()
# scikit-learnのインポートをします

from sklearn import tree



target = train["Survived"].values

features_one = train[["Pclass", "Sex_female", "Sex_male", "Age", "Fare"]].values

 

# 決定木の作成

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

 

# 「test」の説明変数の値を取得

test_features = test[["Pclass", "Sex_female", "Sex_male", "Age", "Fare"]].values

 

# 「test」の説明変数を使って「my_tree_one」のモデルで予測

my_prediction = my_tree_one.predict(test_features)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

 

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

 

# my_tree_one.csvとして書き出し

my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])










# X_train = train.drop(['Survived'], axis=1)  # X_trainはtrainのSurvived列以外

# y_train = train['Survived']  # Y_trainはtrainのSurvived列
# import lightgbm as lgb

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score



# # X_trainとY_trainをtrainとvalidに分割

# train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.33, random_state=0)



# # lab.Datasetでtrainとvalidを作っておく

# lgb_train = lgb.Dataset(train_x, train_y)

# lgb_eval = lgb.Dataset(valid_x, valid_y)



# # パラメータを定義

# lgbm_params = {'objective': 'binary'}



# # lgb.trainで学習

# evals_result = {}

# gbm = lgb.train(params = lgbm_params,

#                 train_set = lgb_train,

#                 valid_sets= [lgb_train, lgb_eval],

#                 early_stopping_rounds=20,

#                 evals_result=evals_result,

#                 verbose_eval=10);
# # valid_xについて推論

# oof = (gbm.predict(valid_x) > 0.5).astype(int)

# print('score', round(accuracy_score(valid_y, oof)*100,2))
# import matplotlib.pyplot as plt



# plt.plot(evals_result['training']['binary_logloss'], label='train_loss')

# plt.plot(evals_result['valid_1']['binary_logloss'], label='valid_loss')

# plt.legend()
# test_pred = (gbm.predict(test) > 0.5).astype(int)

# test_pred

# gender_submission['Survived'] = test_pred

# print(gender_submission)

# gender_submission.to_csv('train_test_split.csv', index=False)

# # trainとtestを縦に連結

# df_full = pd.concat([train, test], axis=0, sort=False)



# print(df_full.shape) # df_fullの行数と列数を確認



# df_full.describe() # df_fullの要約統計量
# # include引数に'O'を指定することで、オブジェクト型の要素数、ユニーク数、最頻値、最頻値の出現回数を表示

# df_full.describe(include='O')
# train.head()
# test.head()
# # dataFrameの可視化

# import pandas_profiling as pdp

# pdp.ProfileReport(train)