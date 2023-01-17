# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You************ can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#トレーニングデータ、テストデータ、サンプルサブミットデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train
train.head(10)
print(train.shape) #学習用データ
print(test.shape) #本番予測用データ
print(gender_submission.shape) #提出データのサンプル
print(train.columns) #トレーニングデータの列名
print('-' * 10) #区切り線を表示
print(test.columns) #テストデータの列名
train.info()
test.info()
test.head()
train.head()
train.isnull().sum()
test.isnull().sum()
#trainとtestを縦に連結
df_full = pd.concat([train, test], axis=0, sort=False)

print(df_full.shape) #df_fullの行数と列数を確認

df_full.describe() #df_fullの要約統計量
#percentilesに10％から90％までを10％刻みで指定
df_full.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
#include引数に'O'を指定することで、オブジェクト型の要素数、ユニーク数、最頻値の出現回数を表示
df_full.describe(include='O')
import pandas as pd
train = pd.read_csv('../input/titanic/train.csv')

#pandas_profilingのインポートとレポートの作成
import pandas_profiling as pdp
pdp.ProfileReport(train)
!pip install japanize-matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
sns.countplot(x='Survived', data=train)
plt.title('死亡者と生存者の数')
plt.xticks([0, 1],['死亡者数', '生存者数'])
plt.show()

#死亡者数と生存者数を表示する
display(train['Survived'].value_counts())

#死亡者数と生存者割合を表示する
display(train['Survived'].value_counts()/len(train['Survived']))
#男女別の生存者数を可視化
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('男女別の死亡者と生存者の数')
plt.legend(['死亡', '生存'])
plt.show()

#SexとSurvivedをクロス集計する
display(pd.crosstab(train['Sex'], train['Survived']))

#クロス集計しSexごとに正規化する
display(pd.crosstab(train['Sex'], train['Survived'], normalize='index'))
#チケットクラス別生存者数を可視化
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('チケットクラス別の死亡者と生存者の数')
plt.legend(['死亡', '生存'])
plt.show()

#PclassとSurvivedをクロス集計する
display(pd.crosstab(train['Pclass'], train['Survived']))

#クロス集計し、Pclassごとに正規化する
display(pd.crosstab(train['Pclass'], train['Survived'], normalize='index'))
#全体のヒストグラム
sns.distplot(train['Age'].dropna(), kde=False, bins=30, label='全体')

#死亡者のヒストグラム
sns.distplot(train[train['Survived']==0].Age.dropna(), kde=False, bins=30, label='死亡')

#生存者ヒストグラム
sns.distplot(train[train['Survived']==1].Age.dropna(), kde=False, bins=30, label='生存')

plt.title('乗船者の年齢の分布')
plt.legend() #凡例を表示
#年齢を8等分し、CategoricalAgeという変数を作成
train['CategoricalAge'] = pd.cut(train['Age'], 8)

#CategoricalAgeとSurvivedをクロス集計する
display(pd.crosstab(train['CategoricalAge'], train['Survived']))

#クロス集計しCategoricalAgeごとに正規化する
display(pd.crosstab(train['CategoricalAge'], train['Survived'], normalize='index'))
sns.countplot(x='SibSp', data=train)
plt.title('同乗している兄弟・配偶者の数')
#SibSpが0か1であればそのまま、2以上であれば2である特徴量SibSp_0_1_2overを作成
train['SibSp_0_1_2over'] = [i if i <= 1 else 2 for i in train['SibSp']]

#SibSp_0_1_2_overごとに集計し、可視化
sns.countplot(x='SibSp_0_1_2over', hue='Survived', data=train)
plt.legend(['死亡', '生存'])
plt.xticks([0, 1, 2], ['0人', '1人', '2人以上'])
plt.title('同乗している兄弟・配偶者の数別の死亡者と生存者の数')
plt.show()

#SibSpとSurvivedをクロス集計する
display(pd.crosstab(train['SibSp_0_1_2over'], train['Survived']))

#クロス集計しSibSpごとに正規化する
display(pd.crosstab(train['SibSp_0_1_2over'], train['Survived'], normalize='index'))
sns.countplot(x='Parch', data=train)
plt.title('同乗している両親・子供の数')
#2以上であればそのままの数、3以上は3という変換を行う
train['Parch_0_1_2_3over']=[i if i<=2 else 3 for i in train['Parch']]

#Parch_0_1_2_3overごとに集計し可視化
sns.countplot(x='Parch_0_1_2_3over', hue='Survived', data=train)
plt.title('同乗している両親・子供の数別の死亡者と生存者の数')
plt.legend(['死亡', '生存'])
plt.xticks([0, 1, 2, 3], ['0人', '1人', '2人', '3人以上'])
plt.xlabel('Parch')
plt.show()

#ParchとSurvivedをクロス集計する
display(pd.crosstab(train['Parch_0_1_2_3over'], train['Survived']))

#クロス集計しParchごとに正規化する
display(pd.crosstab(train['Parch_0_1_2_3over'], train['Survived']))

#クロス集計しParchごとに正規化する
display(pd.crosstab(train['Parch_0_1_2_3over'], train['Survived'],
                    normalize='index'))
#SibSpとParchが同乗している家族の数。1を足すと家族の人数となる
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

#lsAloneを0とし、2行目でFamilySizeが2以上であれば1にしている
train['lsAlone'] = 0
train.loc[train['FamilySize']>=2, 'lsAlone'] = 1

#lsAloneごとに可視化
sns.countplot(x='lsAlone', hue='Survived', data=train)
plt.xticks([0, 1], ['１人','2人以上'])

plt.legend(['死亡', '生存'])
plt.title('１人or２人以上で乗船別の死亡者と生存者の数')
plt.show()

#lsAloneとSurvivedをクロス集計する
display(pd.crosstab(train['lsAlone'], train['Survived']))

#クロス集計しlsAloneごとに正規化する
display(pd.crosstab(train['lsAlone'], train['Survived'], normalize='index'))
sns.distplot(train['Fare'].dropna(), kde=False, hist=True)
plt.title('運賃の分布')
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

#CategoricalFareとSurvivedをクロス集計する
display(pd.crosstab(train['CategoricalFare'], train['Survived']))

#クロス集計し、CategoricalFareごとに正規化する
display(pd.crosstab(train['CategoricalFare'], train['Survived'],
                    normalize='index'))
train['Name'][0:5]
#敬称を抽出し、重複を省く
set(train.Name.str.extract(' ([A-Za-z]+)\.',expand=False))
#敬称をcountする
train.Name.str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()
#trainにtitle列を作成、Title列の値は敬称
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

#trainのtitle列の値ごとに平均値を算出
train.groupby('Title').mean()['Age']
#変換するための関数を作成
def title_to_num(title):
    if title=='Master':
        return 1
    elif title=='Miss':
        return 2
    elif title=='Mr':
        return 3
    elif title=='Mrs':
        return 4
    else:
        return 5
    
#testにもtitle列を作成
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#リスト内包表記を用いて変換
train['Title_num'] = [title_to_num(i) for i in train['Title']]
test['Title_num'] = [title_to_num(i) for i in test['Title']]
        
train = pd.get_dummies(train, columns=['Sex', 'Embarked'])
test = pd.get_dummies(test, columns=['Sex', 'Embarked'])

#不要な列の削除
train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

#trainの表示
display(train.head())
X_train = train.drop(['Survived'], axis=1) #X_trainはtrainのSurvived列以外
y_train = train['Survived'] #y_trainはtrainのSurvived列
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#X_trainとy_trainをtrainとvalidに分割
train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.33, random_state=0)

#lgb.Datasetでtrainとvalidを作っておく
lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(valid_x, valid_y)

#パラメータを定義
lgbm_params = {'objective': 'binary'}

#lgb.trainで学習
evals_result = {}
gbm = lgb.train(params=lgbm_params,
                train_set = lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                early_stopping_rounds=20,
                evals_result=evals_result,
                verbose_eval=10);
