# 前処理に必要なモジュールの読み込み

import pandas as pd
# 読み込んだデータはExcelの表のような形式で扱う（行と列がある）

# モデル作成用データの読み込み（生存か死亡か知っているデータ）

train_df = pd.read_csv('../input/train.csv')

# 予測対象データの読み込み（生存か死亡か知らないデータ）

test_df = pd.read_csv('../input/test.csv')
# モデル作成用データのサイズを確認

# (行数, 列数) で表示される

train_df.shape
# 予測対象データのサイズを確認

# モデル作成用データに対して1列少ない

test_df.shape
# モデル作成用データの上から5行を表示

# 参考: train_df.head(7) # 上から7行表示

train_df.head()
# 予測対象データの上から5行を表示

# Survivedの列（生存か死亡かを表す）がないことが確認できる

test_df.head()
# モデル作成用データの情報を確認

train_df.info()
# 予測対象データの情報を確認

test_df.info()
# ここまでの分析を元に、以下の4つの情報から生死を予測することにする

columns = ['Age', 'Pclass', 'Sex', 'Embarked']
# モデルが予測に使う情報をX, モデルが予測する情報（ここでは生死）をyとする（Xとyという変数名が多い）

X = train_df[columns].copy()

y = train_df['Survived']

# 予測対象データについて、予測に使う情報を取り出しておく

X_test = test_df[columns].copy()
X.head()
# モデル作成用データの欠損値の確認

X.isnull().sum()
# 予測対象データの欠損値の確認

X_test.isnull().sum()
# Ageの欠損を平均値で埋める

# **Note**: もくもくタイムで他の埋め方を試す際は、このセルを置き換えます

age_median = X['Age'].median()

print(f'Age mean: {age_median}')

X['AgeFill'] = X['Age'].fillna(age_median)

X_test['AgeFill'] = X_test['Age'].fillna(age_median)
# 欠損を含むAge列を削除（年齢の情報はAgeFill列を参照する）

X = X.drop(['Age'], axis=1)

X_test = X_test.drop(['Age'], axis=1)
# Embarkedの欠損値を埋める

embarked_freq = X['Embarked'].mode()[0]

print(f'Embarked freq: {embarked_freq}')

X['Embarked'] = X['Embarked'].fillna(embarked_freq)

# X_testにEmbarkedの欠損値がないため、実施しない
# モデル作成用データの欠損値(Embarked, AgeFill)が埋まったことを確認

X.isnull().sum()
# 予測対象データの欠損値が埋まったことを確認

X_test.isnull().sum()
# 性別（female/male）を0/1に変換する（maleとfemaleのままではsklearnが扱えない）

# カテゴリを整数に置き換えるための辞書を用意

gender_map = {'female': 0, 'male': 1}

# 引数の辞書のキー（コロンの左側）に一致する要素が、辞書の値（コロンの右側）に置き換わる（femaleが0に置き換わり、maleが1に置き換わる）

# 注: Sexの取りうる値はfemaleかmale

X['Gender'] = X['Sex'].map(gender_map).astype(int)

X_test['Gender'] = X_test['Sex'].map(gender_map).astype(int)
# Sexに代えてGenderを使うため、Sex列を削除する

X = X.drop(['Sex'], axis=1)

X_test = X_test.drop(['Sex'], axis=1)
# Embarked（S, Q, Cという3カテゴリ）をダミー変数にする

# （Embarked列が消え、Embarked_S, Embarked_Q, Embarked_C列が追加される）

X = pd.get_dummies(X, columns=['Embarked'])

X_test = pd.get_dummies(X_test, columns=['Embarked'])
# 前処理したモデル作成用データの確認

X.head()
# 前処理した予測対象データの確認

X_test.head()
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
# 決定木というアルゴリズムを使ったモデルを用意

model = DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3, min_samples_leaf=2)

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)
# モデル性能確認用データについて生死を予測

pred = model.predict(X_val)

# accuracyを算出して表示

accuracy_score(y_val, pred)
# 予測対象データについて生死を予測

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
# （案1） 中央値で埋める（年齢を大きい順に並べたときに中央に来る値。平均値とは異なる値となることが多い）

"""

age_median = X['Age'].median()

print(f'Age mean: {age_median}')

X['AgeFill'] = X['Age'].fillna(age_median)

X_test['AgeFill'] = X_test['Age'].fillna(age_median)

"""
# (案2) 仮説: 年齢の平均値は性別ごとに違うのでは？

# AgeFill列を作る前に、性別ごとの年齢の平均値を確認

# X[['Sex', 'Age']].groupby('Sex').mean()
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

X['AgeFill'] = X[['Age', 'Sex']].apply(age_by_sex, axis=1)

X_test['AgeFill'] = X_test[['Age', 'Sex']].apply(age_by_sex, axis=1)

"""
# (案3) 仮説: 年齢の平均値はチケットの階級ごとに違うのでは？（年齢高い→お金持っている→いいチケット）

# AgeFill列を作る前に、チケットの等級ごとの年齢の平均値を確認

# X[['Pclass', 'Age']].groupby('Pclass').mean()
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

X['AgeFill'] = X[['Age', 'Pclass']].apply(age_by_pclass, axis=1)

X_test['AgeFill'] = X_test[['Age', 'Pclass']].apply(age_by_pclass, axis=1)

"""
"""

# 決定木というアルゴリズムを使ったモデルを用意

model = DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3, min_samples_leaf=2)

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)

"""