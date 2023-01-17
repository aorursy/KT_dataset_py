# Shift+Enterで実行できます

# 詳しくは藤原さんの勉強会資料を参考にしてください

## https://extra-confluence.gree-office.net/pages/viewpage.action?pageId=170275104



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# データセット読み込み

## 学習用データ

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

## テスト用データ

df_test  = pd.read_csv('/kaggle/input/titanic/test.csv')

## 提出サンプル

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# データセットの先頭5件確認

df_train.head()
df_test.head()
submission.head()
# NaN(=欠損値)があると正しく学習できないため、欠損値を0で埋める

train = df_train.fillna(0)

## テストデータにも同じ処理を加える

test = df_test.fillna(0)
## こんな感じ

train.head()
# trainには目的変数(=生きてるか死んでるか)である'Survived'があるのでそれをモデルの学習のために分けます

## 分け方はなんでもいいんですがわかりやすいように以前の抽出法と揃えます

data_x = train.loc[:,['PassengerId', 'Pclass', 'Name','Sex', 'Age','SibSp', 'Parch', 'Ticket', 'Fare','Cabin','Embarked']]

data_y = train.loc[:,['Survived']]
# Name, Sex, Ticket, Cabin, Embarkedは数字以外のものが入っていると学習モデルは混乱してしまうので一旦捨てます

data_x = data_x.loc[:,['PassengerId', 'Pclass', 'Age','SibSp', 'Parch', 'Fare']]



## テストデータにも同様の処理

## テストデータは目的変数は入ってません、それをこれから予測するので

test = test.loc[:,['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
# さらに学習用データをモデル学習用とモデル検証用データに分けます

## 分けるのに便利なライブラリ読み込み

from sklearn.model_selection import train_test_split

# 引数をつけなければ、モデル学習データ75％、検証データ25％に分けます

## シード値を指定しないと実行するたびランダムに分かれます

train_x,valid_x,train_y,valid_y = train_test_split(data_x,data_y)
# 以前触った「勾配ブースティング木」モデルを使って学習してみます

## 読み込み

from sklearn.ensemble import GradientBoostingClassifier

# モデルを作ります。引数でハイパーパラメータのチューニングができますが、いったんデフォルトで大丈夫です

gb = GradientBoostingClassifier()

# モデルをトレーニングします

gb.fit(train_x, train_y)
# 予測値をみてみます

print(gb.predict(valid_x))


# 正解率を確認します

# 学習に使っていないテストデータで、学習の結果を確認します

print(gb.score(valid_x,valid_y)*100)
# 正答率約70%のモデルを使って、テストデータから予測してみます

target = gb.predict(test)
# targetには予測した答えが入っているので提出しましょう

# 結果をCSVに変換

sample_submit = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": target

})

sample_submit.to_csv('submission.csv', index = False)   
## おまけ

print(df_train['Sex'])
print(pd.get_dummies(df_train['Sex']))