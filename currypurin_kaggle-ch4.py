# warningsを無視する
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib 
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_gender_submission = pd.read_csv("../input/gender_submission.csv")
# 本文にはない、レイアウト設定用
# sns.set_palette("Blues_r", 3) # 青３色のスタイル


# fontsizeの設定
plt.rcParams["font.size"] = 18

# サイズの設定
plt.rcParams['figure.figsize'] = (8.0, 6.0)
df_train.head(5)
print(df_train.shape) # 学習用データ
print(df_test.shape) # 本番予測用データ
print(df_gender_submission.shape) # 提出データのサンプル
print(df_train.columns) # トレーニングデータの列名
print('-'*10) # 区切りを挿入
print(df_test.columns) # テストデータの列名
df_train.info()
df_test.info()
df_train.head()
df_train.isnull().sum() 
# isnull()は、欠損値に対しTrueを返し、欠損値以外にはFalseを返す
# sum()は、Trueを1、Falseを0として合計する
# よってdf.isnull().sum()で欠損値を算出することができる
df_test.isnull().sum()
# df_trainとdf_Testを縦に連結
df_full = pd.concat([df_train, df_test], axis = 0, ignore_index=True)

print(df_full.shape) # df_fullの行数と列数を確認

df_full.describe() # df_fullの要約統計量
df_full.describe(include = 'all')
df_full.describe(include=['O'])
df_full.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
sns.countplot(x='Survived', data=df_train)
plt.title('死亡者と生存者の数')
plt.xticks([0,1],['死亡者', '生存者'])
plt.show()

# 死亡者と生存者数を表示する
display(df_train['Survived'].value_counts())

# 死亡者と生存者割合を表示する
display(df_train['Survived'].value_counts()/len(df_train['Survived']))
# 男女別の生存者数を可視化
sns.countplot(x='Sex', hue='Survived', data=df_train)
# plt.xticks([0.0,1.0], ['死亡','生存'])
# plt.title('男女別の死亡者と生存者の数')
# plt.show()

plt.title('男女別の死亡者と生存者の数')
plt.legend(['死亡','生存'])
plt.show()

# SexとSurvivedをクロス集計する
display(pd.crosstab(df_train['Sex'], df_train['Survived']))

# クロス集計しSexごとに正規化する
display(pd.crosstab(df_train['Sex'], df_train['Survived'],normalize = 'index'))
# チケットクラス別の生存者数を可視化
sns.countplot(x='Pclass', hue='Survived', data=df_train)
plt.title('チケットクラス別の死亡者と生存者の数')
plt.legend(['死亡','生存'])
plt.show()

# PclassとSurvivedをクロス集計する
display(pd.crosstab(df_train['Pclass'], df_train['Survived']))

# クロス集計しPclassごとに正規化する
display(pd.crosstab(df_train['Pclass'], df_train['Survived'],normalize = 'index'))
sns.set_palette('pastel')
# 全体のヒストグラム
sns.distplot(df_train['Age'].dropna(), kde=False, bins = 30 ,label='全体')

# 死亡者のヒストグラム
sns.distplot(df_train[df_train['Survived'] == 0].Age.dropna(), kde = False, bins=30, label='死亡')

# 生存者のヒストグラム
sns.distplot(df_train[df_train['Survived'] == 1].Age.dropna(), kde = False, bins=30, label='生存')

plt.title('乗船者の年齢の分布') # タイトル
plt.legend() # 凡例を表示;
# 年齢を８等分し、CategoricalAgeという変数を作成
df_train['CategoricalAge'] = pd.cut(df_train['Age'], 8)

# CategoricalAgeでグルーピングして、Survivedを平均
df_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
sns.countplot(x='SibSp', data = df_train, color='cornflowerblue')

plt.title("同乗している兄弟・配偶者の数", fontsize = 20);
# SibSpが0か1であればそのまま、2以上であれば2である特徴量SibSp_0_1_2overを作成
df_train['SibSp_0_1_2over'] = [i if i <=1 else 2 for i in df_train['SibSp']]

# SibSp_0_1_2overごとに集計し、可視化 
sns.countplot(x = 'SibSp_0_1_2over', hue = 'Survived', data = df_train)
plt.legend(['死亡', '生存'])
plt.xticks([0,1,2], ['0人', '1人', '2人以上'])
plt.title('同乗している兄弟・配偶者の数別の死亡者と生存者の数')
plt.show()

# SibSpとSurvivedをクロス集計する
display(pd.crosstab(df_train['SibSp_0_1_2over'], df_train['Survived']))

# クロス集計しSibSpごとに正規化する
display(pd.crosstab(df_train['SibSp_0_1_2over'], df_train['Survived'], normalize = 'index'))
sns.countplot(x='Parch', data = df_train)
plt.title('同乗している両親・子供の数');
# 2以下であればそのままの数、3以上は3という変換を行う
df_train['Parch_0_1_2_3over'] = [i if i <=2 else 3 for i in df_train['Parch']]

# Parch_0_1_2_3overごとに集計し可視化
sns.countplot(x='Parch_0_1_2_3over',hue='Survived', data = df_train)
plt.title('同乗している両親・子供の数別の死亡者と生存者の数')
plt.legend(['死亡','生存'])
plt.xticks([0, 1, 2, 3], ['0人', '1人', '2人', '3人以上'])
plt.xlabel('Parch')
plt.show()

# ParchとSurvivedをクロス集計する
display(pd.crosstab(df_train['Parch_0_1_2_3over'], df_train['Survived']))

# クロス集計しParchごとに正規化する
display(pd.crosstab(df_train['Parch_0_1_2_3over'], df_train['Survived'], normalize = 'index'))
#SibSpとParchが同乗している家族の数。1を足すと家族の人数となる
df_train['FamilySize']=df_train['SibSp']+ df_train['Parch']+ 1

# IsAloneを0とし、2行目でFamilySizeが2以上であれば1にしている
df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize'] >= 2, 'IsAlone'] = 1

# IsAloneごとに可視化
sns.countplot(x='IsAlone', hue = 'Survived', data = df_train)
plt.xticks([0, 1], ['1人', '2人以上'])

plt.legend(['死亡', '生存'])
plt.title('１人or２人以上で乗船別の死亡者と生存者の数')
plt.show()

# IsAloneとSurvivedをクロス集計する
display(pd.crosstab(df_train['IsAlone'], df_train['Survived']))

# クロス集計しIsAloneごとに正規化する
display(pd.crosstab(df_train['IsAlone'], df_train['Survived'], normalize = 'index'));
sns.distplot(df_train['Fare'].dropna(), kde=False, hist=True)
plt.title('運賃の分布');
df_train['CategoricalFare'] = pd.qcut(df_train['Fare'], 4)
df_train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

# CategoricalFareとSurvivedをクロス集計する
display(pd.crosstab(df_train['CategoricalFare'], df_train['Survived']))

# クロス集計しCategoricalFareごとに正規化する
display(pd.crosstab(df_train['CategoricalFare'], df_train['Survived'], normalize = 'index'))
df_test['Name'][0:5]
# 敬称を抽出し、重複を省く
set(df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))
# collections.Counterを使用して、数え上げる
import collections
collections.Counter(df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))
# df_trainにTitle列を作成、Title列の値は敬称
df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# df_testにTitle列を作成、Title列の値は敬称
df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# df_trainのTitle列の値ごとに平均値を算出
df_train.groupby('Title').mean()['Age']
# 変換するための関数を作成
def title_to_num(title):
    if title == 'Master':
        return 1
    elif title == 'Miss':
        return 2
    elif title == 'Mr':
        return 3
    elif title == 'Mrs':
        return 4
    else:
        return 5

# リスト内包表記を用いて変換
df_train['Title_num'] = [title_to_num(i) for i in df_train['Title']]
df_test['Title_num'] = [title_to_num(i) for i in df_test['Title']]