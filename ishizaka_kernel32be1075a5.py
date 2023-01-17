# ライブラリのインポート

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.linear_model import LogisticRegression



# Jupyter Notebookの中でインライン表示する場合の設定（これが無いと別ウィンドウでグラフが開く）

%matplotlib inline



# データの読み込み

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# トレーニングデータのProfile Reportを作成

# (出力結果が膨大なのでコメントアウト。必要な時だけ実行)

# pandas_profiling.ProfileReport(train)



# テストデータのProfile Reportを作成

# (出力結果が膨大なのでコメントアウト。必要な時だけ実行)

# pandas_profiling.ProfileReport(test)
# データタイプの確認

train.dtypes
print(train.columns)

print(train.loc[:,['Fare', 'Pclass', 'Cabin']])
def preprocess(df):

    # NameからMr/Mrs/Miss/Masterを取り出し

    df["Mr"] = df["Name"].apply(lambda x: x.count("Mr."))

    df["Mrs"] = df["Name"].apply(lambda x: x.count("Mrs."))

    df["Miss"] = df["Name"].apply(lambda x: x.count("Miss."))

    df["Master"] = df["Name"].apply(lambda x: x.count("Master."))

    return df

train = preprocess(train)

test = preprocess(test)

# いらなそうな名前を削除

train=train.drop('Name', axis='columns')

test=test.drop('Name', axis='columns')
print(train)








# 文字列をラベル化した数値に変換する為のライブラリをインポート

from sklearn.preprocessing import LabelEncoder



# データタイプがobjectの列の値をラベル化した数値に変換

lbl = LabelEncoder()



lbl.fit(list(train['Sex'].values) + list(test['Sex'].values))

train['Sex'] = lbl.transform(list(train['Sex'].values))

test['Sex'] = lbl.transform(list(test['Sex'].values))



lbl.fit(list(train['Ticket'].values) + list(test['Ticket'].values))

train['Ticket'] = lbl.transform(list(train['Ticket'].values))

test['Ticket'] = lbl.transform(list(test['Ticket'].values))



lbl.fit(list(train['Cabin'].values) + list(test['Cabin'].values))

train['Cabin'] = lbl.transform(list(train['Cabin'].values))

test['Cabin'] = lbl.transform(list(test['Cabin'].values))



lbl.fit(list(train['Embarked'].values) + list(test['Embarked'].values))

train['Embarked'] = lbl.transform(list(train['Embarked'].values))

test['Embarked'] = lbl.transform(list(test['Embarked'].values))
print(train['Cabin'])
# トレーニングデータのNaNの数

train_nan = train.isnull().sum()

train_nan = train_nan[train_nan > 0]

train_nan
# テストデータのNaNの数

test_nan = test.isnull().sum()

test_nan = test_nan[test_nan > 0]

test_nan
# Ageをpclassごとの平均値で埋める



Agedf = pd.DataFrame(train.loc[:,['Age','Pclass']])

Pclass1 = Agedf[Agedf.Pclass == 1]

meanAge1 = Pclass1.mean().Age

Pclass2 = Agedf[Agedf.Pclass == 2]

meanAge2 = Pclass2.mean().Age

Pclass3 = Agedf[Agedf.Pclass == 3]

meanAge3 = Pclass3.mean().Age

train.loc[(train["Pclass"].values == 1) & (train["Age"].isnull()), "Age"] = meanAge1

train.loc[(train["Pclass"].values == 2) & (train["Age"].isnull()), "Age"] = meanAge2

train.loc[(train["Pclass"].values == 3) & (train["Age"].isnull()), "Age"] = meanAge3



Agedf = pd.DataFrame(test.loc[:,['Age','Pclass']])

Pclass1 = Agedf[Agedf.Pclass == 1]

meanAge1 = Pclass1.mean().Age

Pclass2 = Agedf[Agedf.Pclass == 2]

meanAge2 = Pclass2.mean().Age

Pclass3 = Agedf[Agedf.Pclass == 3]

meanAge3 = Pclass3.mean().Age



test.loc[(test["Pclass"].values == 1) & (test["Age"].isnull()), "Age"] = meanAge1

test.loc[(test["Pclass"].values == 2) & (test["Age"].isnull()), "Age"] = meanAge2

test.loc[(test["Pclass"].values == 3) & (test["Age"].isnull()), "Age"] = meanAge3

train_nan = train.isnull().sum()

train_nan = train_nan[train_nan > 0]

train_nan
# サブミット用にIDを退避させておく

train_ID = train['PassengerId']

test_ID = test['PassengerId']



# 学習用にデータを分割する

y_train = train['Survived']

X_train = train.drop(['PassengerId','Survived'], axis=1)

X_test = test.drop('PassengerId', axis=1)



# 欠損値を処理する

Xmat = pd.concat([X_train, X_test])

# 欠損値の少ないカラムのNaNは中央値(median)で埋める

Xmat = Xmat.fillna(Xmat.median())



# check whether there are still nan

Xmat_nan = Xmat.isnull().sum()

Xmat_nan = Xmat_nan[Xmat_nan > 0]

Xmat_nan
print(train.iloc[2:4, :])
# trainデータとtestデータを含んでいるXmatを、再度trainデータとtestデータに分割

X_train = Xmat.iloc[:train.shape[0],:]

X_test = Xmat.iloc[train.shape[0]:,:]



# ランダムフォレストをインポート

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=80, max_features='auto')

rf.fit(X_train, y_train)

print("Training done using Random Forest")



# np.argsort()はソート結果の配列のインデックスを返す。引数の頭に"-"をつけると降順。

# つまり"-rf.feature_importances_"を引数にする事で重要度の高い順にソートした配列のインデックスを返す。

ranking = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()

# use the top 10 features only

X_train = X_train.iloc[:,ranking[:10]]

X_test = X_test.iloc[:,ranking[:10]]

# z-scoreにて標準化

# (値 - 平均) / 標準偏差

# X_train = (X_train - X_train.mean()) / X_train.std()

# X_test = (X_test - X_test.mean()) / X_test.std()
# 正規化　Prof.UMEMURAをパクリスペクト



# 正規化 0～1の値にする

X_train = X_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

X_test = X_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# アルゴリズムにロジスティック回帰を採用

lr = LogisticRegression(C=1000)



# fit関数で学習開始

lr.fit(X_train,y_train)

y_test_pred = lr.predict(X_test)

y_test_pred
# Adaboostなるものをためしてみる

from sklearn.ensemble import AdaBoostClassifier as abc

bdt = abc(random_state=0)



bdt.fit(X_train,y_train)

y_test_ada = bdt.predict(X_test)

y_test_ada
# 次は決定木

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train,y_train)

y_test_dtc = clf.predict(X_test)

y_test_dtc
y_test_data = y_test_pred + y_test_ada + y_test_dtc

y_test_data = y_test_data / 3

print(y_test_data)

y_test_data = y_test_data + 0.5

print(y_test_data)

y_test_data = y_test_data.astype(np.int)

y_test_data
# submission

submission = pd.DataFrame({

    "PassengerId": test_ID,

    "Survived": y_test_pred

})

print(submission)
submission.to_csv('submission.csv', index=False)