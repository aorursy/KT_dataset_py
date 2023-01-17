# Import

# 必要なライブラリのインポート

import numpy as np

import pandas as pd

import os

import sys

import glob

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
# Print files

# 使用ファイルの確認。入力ファイルはディレクトリ"../input/"に配置されている。

print(os.listdir("../"))

print(glob.glob("../*/*"))
# Load input datasets to pandas dataframes

# 入力データセットをpandasのデータフレームに読み込み

train_df = pd.read_table('../input/train.csv',sep=',', index_col=0)   # 学習用データセット

test_df  = pd.read_table('../input/test.csv', sep=',', index_col=0)   # テスト用データセット
# Show information and samples of training dataframe.

# 学習データの情報を表示する

train_df.info()     # データフレームの情報を表示

train_df.head()     # データをサンプル的に表示
# Show information and samples of test dataframe.

# テストデータの情報を表示する

test_df.info()     # データフレームの情報を表示

test_df.head()     # データをサンプル的に表示
'''

(note) Meaning of each column

Colname        Description

====================================================================================

PassengerId    Each passenger's identity number.

Survived       Purpose variable.0=No,1=Yes

Pclass         Ticket class.  1 = 1st, 2 = 2nd, 3 = 3rd

Name           Name.

Sex            Sex.

Age            Age in years.

SibSp          # of siblings / spouses aboard the Titanic.

Parch          # of parents / children aboard the Titanic

Ticket         Ticket number.

Fare           Passenger fare.

Cabin          Cabin number

Embarked       Port of Embarkation.C=Cherbourg, Q=Queenstown, S=Southampton



(備忘) 各カラムの意味を記載しておく

カラム名        カラム説明

====================================================================================

PassengerId    乗客ID

Survived       目的変数。生き残ったかどうか。0=No,1=Yes

Pclass         チケットのクラス。  1=1st, 2=2nd, 3=3rd

Name           名前。

Sex            性別。

Age            年齢。

SibSp          タイタニックに同乗した兄弟/配偶者の数。

Parch          タイタニックに同乗した親/子の数。

Ticket         チケット番号。

Fare           運賃。

Cabin          客室番号。

Embarked       乗船した港。C=シェルブール(仏北西部), Q=クイーンズタウン(愛蘭), S=サザンプトン(英南部)

'''
'''   Quantificate training and test data   '''

'''   データの数値化   '''

# Save index(PassengerId)

# インデックス(乗客ID)を保存する。

train_df_index = train_df.index

test_df_index = test_df.index
# Concatenate training and test data to align the number of columns of eath dataframe after quantification

# ダミー化した後のカラム数を揃えるため、一時的に学習データとテストデータを結合する

tmp_df = pd.concat([train_df, test_df],sort=False)

tmp_df

# Set survived flag of test data to 99999

# 学習データのカラム"Survived"は一時的に99999にしておく

tmp_df['Survived'] = tmp_df['Survived'].fillna(99999)

tmp_df
# Drop unnecessary columns

# 不要カラムのドロップ

drop_list = ['Name', 'Ticket']

tmp_df = tmp_df.drop(drop_list, axis=1)

tmp_df
# Replace the NaN of the room number with a character string, and cut out only the head containing the value

# 客室番号のNaNを文字列に置換し、値が入っているものは先頭のみ切り出す

tmp_df['Cabin'] = tmp_df['Cabin'].fillna('NaN')

tmp_df['Cabin'] = tmp_df['Cabin'].str[:1]
# Replace objects by dummy columns(0 or 1 numeric values)

# 文字列のカラムをダミーカラムで置換する（0か1の数値の値にする）

tmp_df = pd.get_dummies(tmp_df, columns=['Pclass', 'Sex', 'Cabin', 'Embarked'])

tmp_df = tmp_df.fillna(tmp_df.mean())

tmp_df
# Separate training data and test data

# 学習データとテストデータを分離する

test_df  = tmp_df[(tmp_df['Survived'] == 99999)].copy()

test_df  = test_df.drop('Survived', axis=1)

train_df = tmp_df[(tmp_df['Survived'] == 0) | (tmp_df['Survived'] == 1)].copy()
'''   Visualization   '''

'''   可視化   '''

tmp_corr = train_df.corr()

sns.heatmap(tmp_corr, square=True, vmax=1, vmin=-1, center=0)
# Separate explanatory variables and objective variables.

# 説明変数と目的変数を分離する。

train_df_Y = train_df.loc[:,['Survived']].copy()

train_df_X = train_df.drop('Survived', axis=1)

del tmp_df
train_df_Y
test_df
'''   Training   '''

'''   学習   '''

inst = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0,verbose=True)

inst.fit(train_df_X, train_df_Y.values.ravel())
'''   Predicting   '''

'''   予測   '''

pred_result = inst.predict(test_df)

submit_df = pd.DataFrame({"Survived":pred_result}, dtype=int, index=test_df_index)

submit_df.to_csv('submission.csv')