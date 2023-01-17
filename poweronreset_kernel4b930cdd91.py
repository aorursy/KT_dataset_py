import os  #osのディレクトリとか環境変数とかいじれる。

import pandas as pd  #データプロセッシング

import numpy as np  #機械学習のため

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler #機械学習

from sklearn.linear_model import LogisticRegression #ロジスティクス回帰もでる

from sklearn.metrics import accuracy_score

print("start")
#ディレクトリを表示できる。

current_dir = "../input/digit-recognizer"

for dirname, _, filenames in os.walk(current_dir):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#読み込み

train = pd.read_csv('../input/digit-recognizer/train.csv')

print("-------------------------------------")

#中身を表示

print(train.describe)

print("-------------------------------------")
#モデルの適用：：

X = train.copy()

X.drop(["label"], axis=1, inplace=True)

#X = train[cols]

y = train["label"]



print(X)
model = LogisticRegression()

model.fit(X,y)
#学習モデルを使って学習データを予測してみる。

train_predicted = model.predict(X)

print(train_predicted)
#

result = accuracy_score(train_predicted, y)

print(result)
#読み込み

test = pd.read_csv('../input/digit-recognizer/test.csv')

print("-------------------------------------")

#中身を表示

print(test.describe)

print("-------------------------------------")
#モデルの適用：：

X_test = test.copy()

print(X_test.dtypes)

test_predicted = model.predict(X_test)



print(test_predicted)
#予測結果をファイルに保存してコミットする準備を行う

sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
#予測結果を回答ファイルに追記し、新規ファイルとして保存する。

sub["Label"] = list(map(int, test_predicted))

sub.to_csv('submission.csv', index=False)
print("end")