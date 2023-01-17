# 前処理に必要なモジュールの読み込み

import numpy as np

import pandas as pd
# 可視化に必要なモジュールの読み込み

import matplotlib.pyplot as plt



# グラフを埋め込んで表示する指定

%matplotlib inline

# グラフのスタイルの設定（グラフにグリッド線が表示されるようにする）

plt.style.use("ggplot")
# 読み込んだデータはExcelの表のような形式で扱う（行と列がある）

# モデル作成用データの読み込み（生存か死亡か知っているデータ）

train_df = pd.read_csv("../input/train.csv")

# 予測対象データの読み込み（生存か死亡か知らないデータ）

test_df = pd.read_csv("../input/test.csv")
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
# 生死についてヒストグラムを描画(pandasのメソッド)

# ヒストグラム: 区間に含まれるデータの個数を表す。個数を柱の高さに反映させる

# - alpha: ヒストグラムの描画色の透過度

# - kind: 描画するグラフの種類（今回はヒストグラムを指定）

# - bins：ヒストグラムにおけるデータの区間の数（生死が取りうる値は0か1。区間は2つなので、0の数と1の数に分かれる）

train_df["Survived"].plot(alpha=0.6, kind="hist", bins=2)

plt.xlabel("Survived")  # x軸ラベルの設定

plt.ylabel("N")  # y軸ラベルの設定

plt.show()  # これまでに設定したものを描画
# 性別ごとに生死のヒストグラムを表示（ヒストグラムを横に並べて表示する）

# 描画領域と1つ1つのグラフを設定（描画領域はfig、グラフはaxesというリストに入れて一括で扱う）

# 1行2列の描画領域とし、1列目(左側)に男性の生死のヒストグラム、2列目(右側)に女性の生死のヒストグラムを表示

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))



for i, sex in enumerate(["male", "female"]):

    # for文は2回回る。1回目: i=0, sex='male'    2回目: i=1, sex='female'

    # Survived列のうち該当するSexのレコードを抽出し、ヒストグラムを描画

    # 引数axに描画領域中の描画位置を指定（ここでは、axes[0]が左側、axes[1]が右側）

    # → つまり、左側に男性の生死のヒストグラム、右側に女性の生死のヒストグラム

    train_df[train_df["Sex"] == sex]["Survived"].plot(

        alpha=0.6, kind="hist", bins=2, ylim=(0, 500), ax=axes[i]

    )

    axes[i].set_title(sex)



plt.show()
# 年齢の幅

print(f'min age: {train_df["Age"].min()}')

print(f'max age: {train_df["Age"].max()}')
# 年齢のヒストグラム（生存／死亡数の積み上げ）を描画（どの年齢層が助かりやすいのか？）

# 欠損値は描画できないため、一時的に削除する（tmp_dfは欠損値を持たないが、train_dfは欠損値を持つ）

tmp_df = train_df.dropna(subset=["Age"])

# matplotlibのメソッドでヒストグラムを描画

# [死亡者の年齢、生存者の年齢]の順で渡している→同一区間で積み上げて描画される

# - range: ヒストグラムの範囲（0歳〜80歳）

# - stacked: 積み上げの有効無効設定

plt.hist(

    [

        tmp_df[(tmp_df["Survived"] == 0)]["Age"],

        tmp_df[(tmp_df["Survived"] == 1)]["Age"],

    ],

    alpha=0.6,

    range=(0, 80),

    bins=10,

    stacked=True,

    label=("Died", "Survived"),

)

plt.legend(["die", "survived"])

plt.xlabel("Age")

plt.ylabel("N")

plt.show()
# 性別ごとに年齢のヒストグラム（生存／死亡数の積み上げ）を描画

# 欠損値は描画できないため、一時的に削除する（tmp_dfは欠損値を持たないが、train_dfは欠損値を持つ）

tmp_df = train_df.dropna(subset=["Age"])

# 描画領域を用意

fig = plt.figure(figsize=(12, 4))



for i, sex in enumerate(["male", "female"], 1):

    # for文は2回回る。1回目: i=1, sex='male'    2回目: i=2, sex='female'

    # 1回目はヒストグラムを左側に描画、2回目は右側に描画

    ax = fig.add_subplot(1, 2, i)

    # 死亡者と生存者を積み上げたヒストグラム（ヒストグラムの柱の中で、生存／死亡の割合が見て取れる）

    plt.hist(

        [

            tmp_df[(tmp_df["Survived"] == 0) & (tmp_df["Sex"] == sex)]["Age"],

            tmp_df[(tmp_df["Survived"] == 1) & (tmp_df["Sex"] == sex)]["Age"],

        ],

        alpha=0.6,

        range=(0, 80),

        bins=10,

        stacked=True,

        label=("Died", "Survived"),

    )

    ax.set_ylim(0, 120)

    plt.title(sex)

    plt.legend(["die", "survived"])

plt.show()
# チケット等級ごとに生死のヒストグラムを表示（ヒストグラムを横に並べて表示する）

# 1行3列の描画領域とする

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))



for i, pclass in enumerate([1, 2, 3]):

    # for文は3回回る。1回目: i=0, pclass=1    2回目: i=1, pclass=2    3回目: i=0, pclass=3

    # Survived列のうち該当するpclassのレコードを抽出し、ヒストグラムを描画

    # 左側にpclass=1の生死のヒストグラム、中央にpclass=2の生死のヒストグラム、右側にpclass=3の生死のヒストグラム

    train_df[train_df["Pclass"] == pclass]["Survived"].plot(

        alpha=0.6, kind="hist", bins=2, ylim=(0, 400), ax=axes[i]

    )

    axes[i].set_title(f"Pclass {pclass}")



plt.show()
# 参考: 性別／チケット等級／生死の組合せ全てに対して年齢のヒストグラムを描画

tmp_df = train_df.dropna(subset=["Age"])

for pclass in [1, 2, 3]:

    # 描画領域の中に2行2列でヒストグラムを配置するための設定

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[10, 8])

    sex_n = 0  # axesのインデックスとなる

    for sex in ["male", "female"]:

        for survived in [0, 1]:

            # Survived, Sex, Pclassが該当するデータの抽出

            draw_df = tmp_df[

                (

                    (tmp_df["Survived"] == survived)

                    & (tmp_df["Sex"] == sex)

                    & (tmp_df["Pclass"] == pclass)

                )

            ]

            fig = draw_df["Age"].plot(

                alpha=0.6,

                kind="hist",

                bins=10,

                xlim=(0, 80),

                ylim=(0, 70),

                ax=axes[sex_n][survived],

            )  # 該当するデータの範囲内でbinを10等分している

            fig.set_title(f"{sex} pclass={pclass} survived={survived}")

        sex_n += 1

    plt.show()
# 乗船港ごとに生死のヒストグラムを表示（ヒストグラムを横に並べて表示する）

# 1行3列の描画領域とする

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))



for i, embarked in enumerate(["S", "Q", "C"]):

    # for文は3回回る。1回目: i=0, embarked='S'    2回目: i=1, embarked='Q'    3回目: i=0, embarked='C'

    # Survived列のうち該当するembarkedのレコードを抽出し、ヒストグラムを描画

    # 左側にpclass=1の生死のヒストグラム、中央にpclass=2の生死のヒストグラム、右側にpclass=3の生死のヒストグラム

    train_df[train_df["Embarked"] == embarked]["Survived"].plot(

        alpha=0.6, kind="hist", bins=2, ylim=(0, 450), ax=axes[i]

    )

    axes[i].set_title(f"Embarked {embarked}")



plt.show()
# 参考: 性別／チケット等級／生死の組合せ全てに対して年齢のヒストグラムを描画

tmp_df = train_df.dropna(subset=["Age"])

for embarked in ["S", "Q", "C"]:

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[10, 8])

    sex_n = 0

    for sex in ["male", "female"]:

        for survived in [0, 1]:

            # Survived, Sex, Embarkedが該当するデータの抽出

            draw_df = tmp_df[

                (

                    (tmp_df["Survived"] == survived)

                    & (tmp_df["Sex"] == sex)

                    & (tmp_df["Embarked"] == embarked)

                )

            ]

            fig = draw_df["Age"].plot(

                alpha=0.6,

                kind="hist",

                bins=10,

                xlim=(0, 80),

                ylim=(0, 80),

                ax=axes[sex_n][survived],

            )  # 該当するデータの範囲内でbinを10等分している

            fig.set_title(f"{sex} embarked={embarked} survived={survived}")

        sex_n += 1

    plt.show()
# ここまでの分析を元に、以下の4つの情報から生死を予測することにする

columns = ["Age", "Pclass", "Sex", "Embarked"]
# モデルが予測に使う情報をX, モデルが予測する情報（ここでは生死）をyとする（Xとyという変数名が多い）

X = train_df[columns].copy()

y = train_df["Survived"]

# 予測対象データについて、予測に使う情報を取り出しておく

X_test = test_df[columns].copy()
X.head()
# モデル作成用データの欠損値の確認

X.isnull().sum()
# 予測対象データの欠損値の確認

X_test.isnull().sum()
# Ageの欠損を平均値で埋める

# **Note**: モクモクタイムで他の埋め方を試す際は、このセルを置き換えます

age_mean = X["Age"].mean()

print(f"Age mean: {age_mean}")

X["AgeFill"] = X["Age"].fillna(age_mean)

X_test["AgeFill"] = X_test["Age"].fillna(age_mean)
# 欠損を含むAge列を削除（年齢の情報はAgeFill列を参照する）

X = X.drop(["Age"], axis=1)

X_test = X_test.drop(["Age"], axis=1)
# Embarkedの欠損値を埋める

embarked_freq = X["Embarked"].mode()[0]

print(f"Embarked freq: {embarked_freq}")

X["Embarked"] = X["Embarked"].fillna(embarked_freq)

# X_testにEmbarkedの欠損値がないため、実施しない
# モデル作成用データの欠損値(Embarked, AgeFill)が埋まったことを確認

X.isnull().sum()
# 予測対象データの欠損値が埋まったことを確認

X_test.isnull().sum()
# 性別（female/male）を0/1に変換する（maleとfemaleのままではsklearnが扱えない）

# カテゴリを整数に置き換えるための辞書を用意

gender_map = {"female": 0, "male": 1}

# 引数の辞書のキー（コロンの左側）に一致する要素が、辞書の値（コロンの右側）に置き換わる（femaleが0に置き換わり、maleが1に置き換わる）

# 注: Sexの取りうる値はfemaleかmale

X["Gender"] = X["Sex"].map(gender_map).astype(int)

X_test["Gender"] = X_test["Sex"].map(gender_map).astype(int)
# Sexに代えてGenderを使うため、Sex列を削除する

X = X.drop(["Sex"], axis=1)

X_test = X_test.drop(["Sex"], axis=1)
# Embarked（S, Q, Cという3カテゴリ）をダミー変数にする

# （Embarked列が消え、Embarked_S, Embarked_Q, Embarked_C列が追加される）

X = pd.get_dummies(X, columns=["Embarked"])

X_test = pd.get_dummies(X_test, columns=["Embarked"])
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

X_train, X_val, y_train, y_val = train_test_split(

    X, y, test_size=0.3, random_state=1

)
# モデル作成用のデータの数の確認

len(y_train)
# モデル性能確認用のデータの数の確認

len(y_val)
# ロジスティック回帰というアルゴリズムを使ったモデルを用意

model = LogisticRegression(random_state=1, solver="liblinear")

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)
# モデル性能確認用データについて生死を予測

pred = model.predict(X_val)

# accuracyを算出して表示

accuracy_score(y_val, pred)
# 予測対象データについて生死を予測

pred = model.predict(X_test)
# 提出用データの形式に変換

submission = pd.DataFrame(

    {"PassengerId": test_df["PassengerId"], "Survived": pred}

)

# 提出用データ作成

submission.to_csv("submission.csv", index=False)
# 以下のコードはお手元では実行不要です

# import pandas as pd

# gender_submission_df = pd.read_csv("../input/gender_submission.csv")

# gender_submission_df.to_csv("submission.csv", index=False)
# （案1） 中央値で埋める（年齢を大きい順に並べたときに中央に来る値。平均値とは異なる値となることが多い）

"""

age_median = X["Age"].median()

print(f"Age mean: {age_median}")

X["AgeFill"] = X["Age"].fillna(age_median)

X_test["AgeFill"] = X_test["Age"].fillna(age_median)

"""
# (案2) 仮説: 年齢の平均値は性別ごとに違うのでは？

# AgeFill列を作る前に、性別ごとの年齢の平均値を確認

# X[["Sex", "Age"]].groupby("Sex").mean()
# （案2）確認すると、男性の平均年齢 31歳、女性の平均年齢 28歳

'''

def age_by_sex(col):

    """col: [age, sex]と想定"""

    age, sex = col

    if pd.isna(age):  # Ageが欠損の場合の処理

        if sex == "male":

            return 31

        elif sex == "female":

            return 28

        else:  # 整数に変更したsexが含まれる場合など

            print("Sexがmale/female以外の値をとっています")

            return -1

    else:  # Ageが欠損していない場合の処理

        return age





# train_dfからAgeとSexの2列を取り出し、各行についてage_by_sex関数を適用

# age_by_sex関数の返り値でAge列の値を上書きする（欠損の場合は、値が埋められる）

X["AgeFill"] = X[["Age", "Sex"]].apply(age_by_sex, axis=1)

X_test["AgeFill"] = X_test[["Age", "Sex"]].apply(age_by_sex, axis=1)

'''
# (案3) 仮説: 年齢の平均値はチケットの階級ごとに違うのでは？（年齢高い→お金持っている→いいチケット）

# AgeFill列を作る前に、チケットの等級ごとの年齢の平均値を確認

# X[["Pclass", "Age"]].groupby("Pclass").mean()
# （案3） pclass==1 38歳、pclass==2 30歳、pclass==3 25歳

'''

def age_by_pclass(col):

    """col: [age, pclass]と想定"""

    age, pclass = col

    if pd.isna(age):  # Ageが欠損の場合の処理

        if pclass == 1:

            return 38

        elif pclass == 2:

            return 30

        else:  # pclass == 3に相当する

            return 25

    else:  # Ageが欠損していない場合の処理

        return age





X["AgeFill"] = X[["Age", "Pclass"]].apply(age_by_pclass, axis=1)

X_test["AgeFill"] = X_test[["Age", "Pclass"]].apply(age_by_pclass, axis=1)

'''
"""

# 決定木というアルゴリズムを使ったモデルを用意

model = DecisionTreeClassifier(

    random_state=1, criterion="entropy", max_depth=3, min_samples_leaf=2

)

# モデル作成は以下の1行（ここまでの前処理に対してたった1行！）で完了する

model.fit(X_train, y_train)

"""