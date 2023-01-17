a = 1

b = '2'

c = a + 3

d = b + '4'
print(a, b, c, d)
x = [1, 2, 4, 8, 16]

y = [i for i in range(3, 9)]  # リスト内包表記



print('x:', x)

print('y:', y)
x[:3]
y[2:4]
m = {'code':1000, 'name':'hoge'}

print(m)
print('コード:', m['code'])

print('名称:', m['name'])
n = 0

for i in range(5):

    n += i

    

    print('+' + str(i), '=', n)

    

print('---')
for i in range(5):

    

    if (i % 2) == 0:

        print(i)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# pandasのread_csv関数でデータを読み込み、DataFrame型として変数に保持させます。

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
# 先頭n件

train.head(5)
# 行数・列数を取得

train.shape
# 行数取得

# train.shape[0] と同じ

print(len(train))



# 列数取得

# train.shape[1] と同じ

print(len(train.columns))
# 指定カラムのみ（Seriesとして取得）

pid = train['PassengerId']



print(type(pid))

pid.head(5)
# 指定カラムのみ(DataFrameとして取得)

sub_df = train[['Survived','Pclass','Sex','Age']]



print(type(sub_df))

sub_df.head(5)
# 指定カラムのみ(DataFrameとして取得)

sub_df = train[['PassengerId']]



print(type(sub_df))

sub_df.head(5)
# 有効値の数、型情報

train.info()
# 欠損値の数

train.isna().sum()
# 統計量表示

train.describe()
# 統計量を個別に取得

train['Fare'].min(), train['Fare'].max(), train['Fare'].mean(), train['Fare'].median()
# 統計量を個別に取得（最頻値）

train['Pclass'].mode()
# 値リスト

train['Pclass'].unique()
# 値と件数

train['Pclass'].value_counts()
# 条件指定のフィルタリング

train[train['Survived'] == 1].head()
# ヒストグラム … 縦軸に度数、横軸に階級をとった統計グラフ。分布状況の確認に使用。

train['Age'].hist(bins=20) # bins:棒の数
train[train['Survived'] == 0]['Age'].hist(bins=20, alpha=0.3, color='red')

train[train['Survived'] == 1]['Age'].hist(bins=20, alpha=0.3, color='blue')
# 複数プロットしたい場合、plt.subplots(行数,列数) で取得したax行列を使用する

fig, ax = plt.subplots(1, 2, figsize=(12,4))



# ヒストグラム（範囲内での頻度）

train['Pclass'].hist(ax=ax[0])



# カテゴリ単位の棒グラフ

train['Pclass'].value_counts().plot.bar(ax=ax[1])
# 同上

# plt.figure(figsize=(表示横サイズ,表示縦サイズ))

# plt.subplot(行数, 列数, 何番目のプロットか)



plt.figure(figsize=(12,4))



plt.subplot(1, 2, 1)

train['Pclass'].hist()



plt.subplot(1, 2, 2)

train['Pclass'].value_counts().plot.bar()
# クロス集計

pd.crosstab(train['Survived'], train['Pclass'])
fig, ax = plt.subplots(1, 2, figsize=(12,4))



# クロス集計結果の棒グラフ化

pd.crosstab(train['Pclass'], train['Survived']).plot.bar(ax=ax[0])

pd.crosstab(train['Survived'], train['Pclass']).plot.bar(ax=ax[1])
# Seabornを使った 件数の棒グラフ化（hue:集計列名）

sns.countplot(x='Pclass', hue='Survived', data=train)
# 散布図

plt.scatter(train['Age'], train['Fare'], s=6)
%%time



# セルの先頭に"%%time"で、実行時間表示



a = [n for n in range(10^10)]
_='''



複数行コメント（※実際は複数行の文字列）

読み捨て用変数（とりあえず_）に値を設定することで、最終行をコメントとしても出力には表示されない



'''
# 単純に数値変換しても、そこには大小関係は必要ないので、1つだけ1でそれ以外は0のベクトル（行列）に変換

#train = pd.get_dummies(train, columns=['Embarked'])

pd.get_dummies(train, columns=['Embarked']).head(5)
# train['Sex'] = 下記いずれかのコード

train['Sex'].map({'male':0, 'female':1})[:10]
pd.factorize(train['Sex'])[0][:10]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit_transform(train['Sex'])[:10]
# train['Age'] = train['Age'].fillna(0) # <-- 下記inplace=Trueはこれと同じになる

train['Age'].fillna(0, inplace=True)

test['Age'].fillna(0, inplace=True)
# 代表的な方法として、

# 各変数を平均0、標準偏差1に揃える方法（sklearn.preprocessing.StandardScaler）と、

# 各変数を最小値0、最大値1に揃える方法（sklearn.preprocessing.MinMaxScaler）が存在します。



print(train['Age'][:10])



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(train[['Age']])[:10] # 2次元配列のin,out
# 不要な列の削除

drop_columns = ['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

#train.drop(drop_columns, axis=1, inplace=True)

train.drop(drop_columns, axis=1).head(5)
# 予測に使用する列名

cols = ['Pclass','Age']



# 学習用

X_train = train[cols]

y_train = train['Survived']



# 予測用

X_test = test[cols]
from sklearn.linear_model import LogisticRegression



# モデル作成（ロジスティック回帰）

model = LogisticRegression(solver='liblinear', random_state=42)



# 学習

model.fit(X_train, y_train)

# 予測

y_test = model.predict(X_test)



pd.Series(y_test).value_counts()
from sklearn.ensemble import RandomForestClassifier



# モデル作成（ランダムフォレスト）

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)



# 学習

model.fit(X_train, y_train)

# 予測

y_test = model.predict(X_test)



pd.Series(y_test).value_counts()
from sklearn.model_selection import train_test_split



# X_trainとY_trainをtrainとvalidに分割

train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
from sklearn.model_selection import KFold



folds = KFold(n_splits=3)



for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):

    trn_x, trn_y = X_train.iloc[trn_], y_train.iloc[trn_]

    val_x, val_y = X_train.iloc[val_], y_train.iloc[val_]

    

    # （省略）trn_x, trn_y を使って学習、val_x, val_y を使って検証