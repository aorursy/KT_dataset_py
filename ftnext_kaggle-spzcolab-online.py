# 前処理に必要なモジュールの読み込み

# pandasは表形式のデータを扱うためのモジュール

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
# データの一部を見てみる

# モデル作成用データの上から5行を表示

# 参考: train_df.head(7) # 上から7行表示

train_df.head()
# 予測対象データの上から5行を表示

# Survivedの列（生存か死亡かを表す）がないことが確認できる

test_df.head()
# 各列にどんな種類のデータが入っているか確かめる（数値なのか、文字列なのか）

# モデル作成用データの情報を確認（見方については後述）

train_df.info()
# 予測対象データの情報を確認

test_df.info()
# モデル作成・性能評価に使うモジュールの読み込み

# scikit-learn（コードではsklearn）は、機械学習の様々なアルゴリズムや、モデルの評価ツールを提供するモジュール

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
# モデルが予測に使う情報（ここでは性別）をx, モデルが予測する情報（ここでは生死）をyとして取り出す

x = train_df['Sex']

y = train_df['Survived']
# xの値のうち、femaleが1に置き換わり、maleが0に置き換わる（コロンの左側に一致する要素が、コロンの右側の値に置き換わる）

# 注: Sexの取りうる値はfemaleかmale

# astype(int)でデータの型が文字列から整数へ変わることに対応

y_pred = x.map({'female': 1, 'male': 0}).astype(int)
# 予測y_predを実際の生死yで採点し、予測の正解率(accuracy)を表示

accuracy_score(y, y_pred)
# 予測対象データについて生死を予測する

# 予測対象データのSex列の取り出し

x_test = test_df['Sex']

# Sexの値を元に、生死を予測

y_test_pred = x_test.map({'female': 1, 'male': 0}).astype(int)
# 提出用データの形式に変換

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': y_test_pred

})

# 提出用データ作成

submission.to_csv('submission.csv', index=False)
# SexとPclassから生死を予測するモデルを作ることにする

columns = ['Pclass', 'Sex']
# モデル作成に使う情報をX, モデルが予測する情報（ここでは生死）をyとして取り出す（Xとyという変数名が多い）

X = train_df[columns].copy()

y = train_df['Survived']

# 予測対象データについて、予測に使う情報を取り出しておく

X_test = test_df[columns].copy()
# モデル作成に使うデータを確認

X.head()
# 性別（female/male）を0/1に変換する（maleとfemaleのままではモデル作成時に扱えない）

# カテゴリを整数に置き換えるための「辞書」を用意

gender_map = {'female': 0, 'male': 1}

# 引数の辞書でコロンの左側（キー）に一致する要素が、コロンの右側の値に置き換わる（femaleが0に置き換わり、maleが1に置き換わる）

X['Gender'] = X['Sex'].map(gender_map).astype(int)

X_test['Gender'] = X_test['Sex'].map(gender_map).astype(int)
# Sexに代えてGenderを使うため、Sex列を削除する

X = X.drop(['Sex'], axis=1)

X_test = X_test.drop(['Sex'], axis=1)
# モデル作成に使うデータについて、前処理した後の状態を確認

X.head()
# 今回のハンズオンは7:3に分けて進める

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# 学習用データの数の確認

len(y_train)
# 性能評価用のデータの数の確認

len(y_val)
# 決定木というアルゴリズムを使ったモデルを用意

model = DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3, min_samples_leaf=2)

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)
# 性能評価用データについて生死を予測

pred = model.predict(X_val)

# 性能確認:accuracyを算出して表示

accuracy_score(y_val, pred)
# X_testについて生死を予測（予測対象データからSexとPclassをX_testとして取り出し、Xと同様の前処理を行っている）

pred = model.predict(X_test)
# このセルを実行する際に、#submission.to_csv(...)の先頭の#を消してください

# 提出用データの形式に変換

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': pred

})

# 提出用データ作成

#submission.to_csv('submission.csv', index=False) # 先頭の#を消してください
# モデルの予測に使う情報にAgeを追加

columns = ['Pclass', 'Sex', 'Age']
# モデル作成に使う情報をX, モデルが予測する情報（ここでは生死）をyとして取り出す

X = train_df[columns].copy()

y = train_df['Survived']

# 予測対象データについて、予測に使う情報を取り出しておく

X_test = test_df[columns].copy()
# モデル作成に使うデータを確認

X.head()
# モデル作成に使うデータの欠損値の確認

X.isnull().sum()
# モデルが予測するデータの欠損値の確認

X_test.isnull().sum()
# Ageの平均値の算出

age_mean = X['Age'].mean()

print(f'Age mean: {age_mean}')
# 平均値を小数第2位で四捨五入して使う(round関数)

# Ageの欠損を平均値で埋めた列AgeFillを追加

X['AgeFill'] = X['Age'].fillna(round(age_mean, 1))

X_test['AgeFill'] = X_test['Age'].fillna(round(age_mean, 1))
# 欠損を含むAge列を削除（年齢の情報はAgeFill列を参照する）

X = X.drop(['Age'], axis=1)

X_test = X_test.drop(['Age'], axis=1)
# モデル作成に使うデータの欠損値の確認

X.isnull().sum()
# モデルが予測するデータの欠損値の確認

X_test.isnull().sum()
# 性別（female/male）を0/1に変換する（「2.モデル作成を一緒に体験」と同様）

gender_map = {'female': 0, 'male': 1}

X['Gender'] = X['Sex'].map(gender_map).astype(int)

X_test['Gender'] = X_test['Sex'].map(gender_map).astype(int)
# Sexに代えてGenderを使うため、Sex列を削除する

X = X.drop(['Sex'], axis=1)

X_test = X_test.drop(['Sex'], axis=1)
# モデル作成に使うデータについて、前処理した後の状態を確認

X.head()
# 今回のハンズオンではモデル作成用データを7:3に分けて進める

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# 学習用データの数の確認

len(y_train)
# 性能評価用のデータの数の確認

len(y_val)
# 決定木というアルゴリズムを使ったモデルを用意

model = DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3, min_samples_leaf=2)

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)
# 性能評価用データについて生死を予測

pred = model.predict(X_val)

# 性能確認：accuracyを算出して表示

accuracy_score(y_val, pred)
# 予測対象データについて生死を予測

pred = model.predict(X_test)
# このセルを実行する際に、#submission.to_csv(...)の先頭の#を消してください

# 提出用データの形式に変換

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': pred

})

# 提出用データ作成

#submission.to_csv('submission.csv', index=False) # 先頭の#を消してください
# モデルの予測に使う情報にEmbarkedを追加

columns = ['Pclass', 'Sex', 'Embarked']
# モデル作成に使う情報をX, モデルが予測する情報（ここでは生死）をyとして取り出す

X = train_df[columns].copy()

y = train_df['Survived']

# 予測対象データについて、予測に使う情報を取り出しておく

X_test = test_df[columns].copy()
# モデル作成に使うデータを確認

X.head()
# モデル作成に使うデータの欠損値の確認

X.isnull().sum()
# モデルが予測するデータの欠損値の確認

X_test.isnull().sum()
# 一番多くの人が乗っている港の取得

embarked_freq = X['Embarked'].mode()[0]

print(f'Embarked freq: {embarked_freq}')
# Embarkedの欠損を平均値で埋めた列EmbarkedFillを追加

X['EmbarkedFill'] = X['Embarked'].fillna(embarked_freq)

X_test['EmbarkedFill'] = X_test['Embarked'].fillna(embarked_freq)
# 欠損を含むEmbarked列を削除（乗船した港の情報はEmbarkedFill列を参照する）

X = X.drop(['Embarked'], axis=1)

X_test = X_test.drop(['Embarked'], axis=1)
# モデル作成に使うデータの欠損値の確認

X.isnull().sum()
# 性別（female/male）を0/1に変換する（「2.モデル作成を一緒に体験」と同様）

gender_map = {'female': 0, 'male': 1}

X['Gender'] = X['Sex'].map(gender_map).astype(int)

X_test['Gender'] = X_test['Sex'].map(gender_map).astype(int)
# Sexに代えてGenderを使うため、Sex列を削除する

X = X.drop(['Sex'], axis=1)

X_test = X_test.drop(['Sex'], axis=1)
# EmbarkedFill（S, Q, Cという3カテゴリ）をダミー変数にする

# （EmbarkedFill列が消え、EmbarkedFill_S, EmbarkedFill_Q, EmbarkedFill_C列が追加される）

X = pd.get_dummies(X, columns=['EmbarkedFill'])

X_test = pd.get_dummies(X_test, columns=['EmbarkedFill'])
# モデル作成に使うデータについて、前処理した後の状態を確認

X.head()
# 今回のハンズオンではモデル作成用データを7:3に分けて進める

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# 学習用データの数の確認

len(y_train)
# 性能評価用のデータの数の確認

len(y_val)
# 決定木というアルゴリズムを使ったモデルを用意

model = DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3, min_samples_leaf=2)

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)
# 性能評価用データについて生死を予測

pred = model.predict(X_val)

# 性能確認：accuracyを算出して表示

accuracy_score(y_val, pred)
# 予測対象データについて生死を予測

pred = model.predict(X_test)
# このセルを実行する際に、#submission.to_csv(...)の先頭の#を消してください

# 提出用データの形式に変換

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': pred

})

# 提出用データ作成

#submission.to_csv('submission.csv', index=False) # 先頭の#を消してください
# train_df, test_dfからSex, Pclass, Age, Embarkedを取り出す
# Ageの欠損への対応
# Embarkedの欠損への対応
# がテゴリ変数Sexへの対応
# カテゴリ変数Embarkedへの対応
# モデル作成
# 性能確認
# 提出用データ作成
# モデルの予測に使う情報にAgeを追加

columns = ['Pclass', 'Sex', 'Age']
# モデル作成に使う情報をX, モデルが予測する情報（ここでは生死）をyとして取り出す

X = train_df[columns].copy()

y = train_df['Survived']

# 予測対象データについて、予測に使う情報を取り出しておく

X_test = test_df[columns].copy()
# モデル作成に使うデータを確認

X.head()
# モデル作成に使うデータの欠損値の確認

X.isnull().sum()
# モデルが予測するデータの欠損値の確認

X_test.isnull().sum()
# Ageの平均値の算出

age_mean = X['Age'].mean()

print(f'Age mean: {age_mean}')
# 平均値を小数第2位で四捨五入して使う(round関数)

# Ageの欠損を平均値で埋めた列AgeFillを追加

X['AgeFill'] = X['Age'].fillna(round(age_mean, 1))

X_test['AgeFill'] = X_test['Age'].fillna(round(age_mean, 1))
# 欠損を含むAge列を削除（年齢の情報はAgeFill列を参照する）

X = X.drop(['Age'], axis=1)

X_test = X_test.drop(['Age'], axis=1)
# モデル作成に使うデータの欠損値の確認

X.isnull().sum()
# モデルが予測するデータの欠損値の確認

X_test.isnull().sum()
# AgeFillを最小と最大の間で10分割

age_band = pd.cut(X['AgeFill'], 10)

# 区間に含まれる年代の順に表示

age_band.value_counts()
# 年代を若い順に0,1,2,...,9と設定

for df in [X, X_test]:

    df.loc[df['AgeFill'] <= 8.378, 'AgeFill'] = 0

    df.loc[(df['AgeFill'] > 8.378) & (df['AgeFill'] <= 16.336), 'AgeFill'] = 1

    df.loc[(df['AgeFill'] > 16.336) & (df['AgeFill'] <= 24.294), 'AgeFill'] = 2

    df.loc[(df['AgeFill'] > 24.294) & (df['AgeFill'] <= 32.252), 'AgeFill'] = 3

    df.loc[(df['AgeFill'] > 32.252) & (df['AgeFill'] <= 40.21), 'AgeFill'] = 4

    df.loc[(df['AgeFill'] > 40.21) & (df['AgeFill'] <= 48.168), 'AgeFill'] = 5

    df.loc[(df['AgeFill'] > 48.168) & (df['AgeFill'] <= 56.126), 'AgeFill'] = 6

    df.loc[(df['AgeFill'] > 56.126) & (df['AgeFill'] <= 64.084), 'AgeFill'] = 7

    df.loc[(df['AgeFill'] > 64.084) & (df['AgeFill'] <= 72.042), 'AgeFill'] = 8

    df.loc[df['AgeFill'] > 72.042, 'AgeFill'] = 9

    df['AgeFill'] = df['AgeFill'].astype(int) # floatからintに変更
# 性別（female/male）を0/1に変換する（「2.モデル作成を一緒に体験」と同様）

gender_map = {'female': 0, 'male': 1}

X['Gender'] = X['Sex'].map(gender_map).astype(int)

X_test['Gender'] = X_test['Sex'].map(gender_map).astype(int)
# Sexに代えてGenderを使うため、Sex列を削除する

X = X.drop(['Sex'], axis=1)

X_test = X_test.drop(['Sex'], axis=1)
# モデル作成に使うデータについて、前処理した後の状態を確認

X.head()
# 今回のハンズオンではモデル作成用データを7:3に分けて進める

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
# 学習用データの数の確認

len(y_train)
# 性能評価用のデータの数の確認

len(y_val)
# 決定木というアルゴリズムを使ったモデルを用意

model = DecisionTreeClassifier(random_state=1, criterion='entropy', max_depth=3, min_samples_leaf=2)

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)
# 性能評価用データについて生死を予測

pred = model.predict(X_val)

# 性能確認：accuracyを算出して表示

accuracy_score(y_val, pred)
# 予測対象データについて生死を予測

pred = model.predict(X_test)
# このセルを実行する際に、#submission.to_csv(...)の先頭の#を消してください

# 提出用データの形式に変換

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': pred

})

# 提出用データ作成

#submission.to_csv('submission.csv', index=False) # 先頭の#を消してください