# coding:utf8
import numpy as np #numpyのインポート
import pandas as pd #pandasのインポート
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns #seaborn(描画ライブラリ)
import collections

from subprocess import check_output
#print(check_output(["ls","../input"]).decode("utf8"))

#df_train,df_testとしてそれぞれ読み込み
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_gender_submission = pd.read_csv('../input/gender_submission.csv')

#df_train.head(5)

# データフレームの行数と列数の確認
# print(df_train.shape) #学習用データ
# print(df_test.shape) #本番予測用データ
# print(df_gender_submission.shape) #提出データのサンプル

# 
# print(df_train.columns)
# print('-'*10) #区切りを挿入
# print(df_test.columns)

# df_train.info()
# print('-'*10)
# df_test.info()

# 欠損値
df_train.isnull().sum()
df_test.isnull().sum()

# df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)
# print(df_full.shape)
# df_full.describe()

#死亡者と生存者の可視化
# sns.countplot(x='Survived', data=df_train)
# plt.title('死亡者と生存者の数')
# plt.xticks([0,1],['死亡者', '生存者'])

# # Survived列の集計
# #　日本語の文字化け対応が残課題
# df_train['Survived'].value_counts()

# 男女別の生存者数を可視化
# sns.countplot(x='Survived', hue='Sex', data=df_train)
# plt.xticks([0.0,1.0],['死亡','生存'])
# plt.title('男女別の死亡者と生存者の数')

#df_train[['Sex','Survived']].groupby(['Sex']).mean()

# # チケットクラス別の生存者数を可視化
# sns.countplot(x='Survived', hue='Pclass', data=df_train)
# plt.xticks([0.0, 1.0],['死亡','生存'])

# #チケットクラス別の生存割合を表示する
# df_train[['Pclass','Survived']].groupby(['Pclass']).mean()

# 年齢の分布
# 全体のヒストグラム
#sns.distplot(df_train['Age'].dropna(),kde=False, bins=30,label='全体')

#死亡者のヒストグラム
#sns.distplot(df_train[df_train['Survived']==0].Age.dropna(),kde=False, bins=30,label='死亡')

#生存者のヒストグラム
#sns.distplot(df_train[df_train['Survived']==1].Age.dropna(),kde=False, bins=30,label='死亡')

# 同乗している兄弟・配偶者の数
#sns.countplot(x='SibSp', data=df_train)

# SibSpが0か1であればそのまま、２以上であれば２である特徴量SibSp_0_1_2overを作成
#df_train['SibSp_0_1_2over']=[i if i<=1 else 2 for i in df_train['SibSp']]

# SibSp_0_1_2overごとに集計し可視化
# sns.countplot(x='Survived', hue='SibSp_0_1_2over', data=df_train)

# plt.legend(['0人','１人','２人以上'], loc= 'upper right')
# plt.title('同乗している兄弟・配偶者の数別の死亡者と生存者の数')

# df_train[['SibSp_0_1_2over','Survived']].groupby(['SibSp_0_1_2over']).mean()

# SibSpとParchが同乗している家族の数。1を足すと家族の人数となる
df_train['FamilySize']=df_train['SibSp']+ df_train['Parch']+1

# IsAloneを0とし、２行目でFamilySizeが１であれば１にしている
df_train['IsAlone']=0
df_train.loc[df_train['FamilySize']==1, 'IsAlone'] = 1

# IsAloneごとに可視化
# sns.countplot(x='Survived',hue='IsAlone',data=df_train)

# plt.legend(['２人以上','１人で乗船'])
# plt.xticks([0,1],['死亡','生存'])

# plt.title('１人or２人以上で乗船別の死亡者と生存者の数')

# 乗船者の運賃の分布
# sns.distplot(df_train['Fare'].dropna(), kde=False, hist=True)
# df_train['CategoricalFare']= pd.qcut(df_train['Fare'], 4)
# df_train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

df_test['Name'][0:5]
#敬称を抽出し、重複を省く
set(df_train.Name.str.extract(' ([A-Za-z]+)\.',expand=False))
collections.Counter(df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False))

df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

df_train.groupby('Title').mean()['Age']

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

df_train['Title_num'] = [title_to_num(i) for i in df_train['Title']]
df_test['Title_num'] = [title_to_num(i) for i in df_test['Title']]