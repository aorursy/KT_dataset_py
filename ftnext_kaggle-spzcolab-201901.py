# 前処理に必要なモジュールの読み込み
import numpy as np
import pandas as pd
# 可視化に必要なモジュールの読み込み
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# 読み込んだデータはExcelの表のような形式で扱う（行と列がある）
# 学習用データの読み込み（生存か死亡か知っているデータ）
train_df = pd.read_csv('../input/train.csv')
# テスト用データの読み込み（生存か死亡か知らないデータ）
test_df = pd.read_csv('../input/test.csv')
# 学習用データのサイズを確認
# (行数, 列数) で表示される
train_df.shape
# テスト用データのサイズを確認
# 学習用データに対して1列少ない
test_df.shape
# 学習用データの上から5行を表示
# 参考: train_df.head(7) # 上から7行表示
train_df.head()
# テスト用データの上から5行を表示
# Survivedの列（生存か死亡かを表す）がないことが確認できる
test_df.head()
# 学習用データの情報を確認
train_df.info()
# テスト用データの情報を確認
test_df.info()
# 学習データについて欠けたデータがある列を確認（infoの情報に、891よりも少ない数の列があった）
train_df.isnull().sum()
# テスト用データについて欠けたデータがある列を確認
test_df.isnull().sum()
# 学習用データの値を使って欠損を埋めるために使う値を表示
train_df[['Age', 'Fare', 'Embarked']].describe([.5], 'all')
# 参考: Ageの分布（欠損値を除いて描画）
sns.distplot(train_df['Age'].dropna(), kde=False, bins=20)
plt.show()
# 参考: Fareの分布
sns.distplot(train_df['Fare'], kde=False, bins=50)
plt.show()
# Ageの欠損を平均値 30歳 で埋める
# **Note**: モクモクタイムで他の埋め方を試す際は、このセルを置き換えます
train_df['Age'] = train_df['Age'].fillna(30)
test_df['Age'] = test_df['Age'].fillna(30)
# Embarkedの欠損を、一番多い乗船港 S で埋める
train_df['Embarked'] = train_df['Embarked'].fillna('S')
# Fareの欠損を 中央値 14.4542 で埋める
test_df['Fare'] = test_df['Fare'].fillna(14.4542)
# 学習用データの欠損値が埋まったことを確認
train_df.isnull().sum()
# テスト用データの欠損値が埋まったことを確認
test_df.isnull().sum()
# カテゴリを整数に置き換えるための辞書を用意
gender_map = {'female': 1, 'male': 0}
# 引数の辞書のキーに一致する要素が、辞書の値に置き換わる（femaleが1に置き換わり、maleが0に置き換わる）
# 注: Sexの取りうる値はfemaleかmale
train_df['Sex'] = train_df['Sex'].map(gender_map)
test_df['Sex'] = test_df['Sex'].map(gender_map)
# Embarked（S, Q, Cという3カテゴリ）をダミー変数にする
train_df = pd.get_dummies(train_df, columns=['Embarked'])
test_df = pd.get_dummies(test_df, columns=['Embarked'])
# 取り除く列のリスト
not_use_columns = ['Name', 'Ticket', 'Cabin']
# 学習用データから列を削除する（PassengerIdは後ほど取り除く）
train_df.drop(not_use_columns, axis=1, inplace=True)
# テスト用データから列を削除する
test_df.drop(not_use_columns, axis=1, inplace=True)
# 前処理した学習用データの確認
train_df.head()
# 前処理したテスト用データの確認
test_df.head()
# 慣例にのっとり、モデルが予測に使うデータをX, モデルが予測するデータをyとする
X = train_df.drop(['PassengerId', 'Survived'], axis=1)
y = train_df['Survived']
# モデル作成・性能評価に使うモジュールの読み込み
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
# 今回のハンズオンは7:3に分けて進める
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# モデル作成用のデータの数の確認
len(y_train)
# モデル性能確認用のデータの数の確認
len(y_val)
# ロジスティック回帰というアルゴリズムを使ったモデルを用意
model = LogisticRegression(random_state=1, solver='liblinear')
# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する
model.fit(X_train, y_train)
# モデル性能確認用データについて生死を予測
pred = model.predict(X_val)
# accuracyを算出して表示
accuracy_score(y_val, pred)
# テスト用データからPassengerId列を除く
X_test = test_df.drop(['PassengerId'], axis=1)
# テスト用データについて生死を予測
pred = model.predict(X_test)
# 提出用データの形式に変換
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': pred
})
# 提出用データ作成
submission.to_csv('submission.csv', index=False)
# 以下のコードはお手元では実行不要です
# import pandas as pd
# gender_submission_df = pd.read_csv('../input/gender_submission.csv')
# gender_submission_df.to_csv('submission.csv', index=False)
# （案1） 中央値 28歳 で埋める
"""
train_df['Age'] = train_df['Age'].fillna(28)
test_df['Age'] = test_df['Age'].fillna(28)
"""
# (案2) 仮説: 年齢の平均値は性別ごとに違うのでは？
# 性別ごとの年齢の平均値を確認
# train_df[['Sex', 'Age']].groupby('Sex').mean()
# （案2）確認すると、男性の平均年齢 31歳、女性の平均年齢 28歳
"""
def age_by_sex(col):
    '''col: [age, sex]と想定'''
    age, sex = col
    if pd.isna(age): # Ageが欠損の場合の処理
        if sex == 'male':
            return 31
        elif sex == 'female':
            return 28
        else: # 整数に変更したsexが含まれる場合など
            print('Sexがmale/female以外の値をとっています')
            return -1
    else: # Ageが欠損していない場合の処理
        return age
# train_dfからAgeとSexの2列を取り出し、各行についてage_by_sex関数を適用
# age_by_sex関数の返り値でAge列の値を上書きする（欠損の場合は、値が埋められる）
train_df['Age'] = train_df[['Age', 'Sex']].apply(age_by_sex, axis=1)
test_df['Age'] = test_df[['Age', 'Sex']].apply(age_by_sex, axis=1)
"""
# (案3) 仮説: 年齢の平均値はチケットの階級ごとに違うのでは？（年齢高い→お金持っている→いいチケット）
# チケットの等級ごとの年齢の平均値を確認
# train_df[['Pclass', 'Age']].groupby('Pclass').mean()
# （案3） pclass==1 38歳、pclass==2 30歳、pclass==3 25歳
"""
def age_by_pclass(col):
    '''col: [age, pclass]と想定'''
    age, pclass = col
    if pd.isna(age): # Ageが欠損の場合の処理
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 30
        else: # pclass == 3に相当する
            return 25
    else: # Ageが欠損していない場合の処理
        return age
train_df['Age'] = train_df[['Age', 'Pclass']].apply(age_by_pclass, axis=1)
test_df['Age'] = test_df[['Age', 'Pclass']].apply(age_by_pclass, axis=1)
"""
"""
# 決定木というアルゴリズムを使ったモデルを用意
model = DecisionTreeClassifier(random_state=1)
# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する
model.fit(X_train, y_train)
"""