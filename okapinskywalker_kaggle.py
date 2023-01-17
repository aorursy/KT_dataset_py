import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

# from sklearn.model_selection import GridSearchCV, cross_val_score
# データの読み込み(事前にDatasetに登録しておく)

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

# データ整形(前処理)



# データ量と特徴量の数を確認

print(df_train.shape)



# trainデータを確認(とりあえず30件)

df_train.head(30)
# testデータも同様に確認する



print(df_test.shape)



df_test.head(30)
# 欠損値の有無を確認

print('学習用データ')

print(df_train.isnull().any())
print('テストデータ')

print(df_test.isnull().any())
# 統計量の確認

df_train.describe()
# 各特徴量間の相関関係を確認

df_train.corr()
# 欠損値(NaN)の補完



def fill_na(data):

#     fillnaで欠損値を補完する

    data.Age = data.Age.fillna(data.Age.median()) # Ageの中央値で補完

    data.Embarked = data.Embarked.fillna('S') # Embarkedの欠損値はすべて'S'で補完

    data.Fare = data.Fare.fillna(data.Fare.mean()) # Fareの平均値で補完

    

    return data



df_train = fill_na(df_train)

df_test = fill_na(df_test)

    

# 文字列データを数値に変換



def convert_vector(data):

    data.Sex = data.Sex.replace(['male', 'female'], [0, 1])

    data.Embarked = data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

    

    return data



df_train = convert_vector(df_train)

df_test = convert_vector(df_test)

# 整形後のデータを確認

df_train.head(30)

# データと各統計量から、学習に使用するカラムを選定

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# DecisionTree(決定木)の実装



# DecisionTreeをインスタンス化。パラメータを指定(しなくても良い)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=7)



# 学習(第1引数に訓練データ、第2引数に正解ラベルを与える)

tree.fit(df_train[predictors], df_train['Survived'])



# 予測

prediction = tree.predict(df_test[predictors])
# 提出用データの作成

df_out = pd.read_csv('../input/test.csv', encoding='utf-8')

df_out['Survived'] = prediction

df_out[['PassengerId', 'Survived']].to_csv('submission_xxxx.csv', index=False)