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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
train_titanic = pd.read_csv("/kaggle/input/titanic/train.csv")
test_titanic = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train_titanic.head()
test_titanic.head()
gender_submission.head()
print(train_titanic.shape)
print(test_titanic.shape)
print(gender_submission.shape)
train_titanic.isnull().sum()
train_titanic.info()
train_titanic.describe()
test_titanic.isnull().sum()
df_full = pd.concat([train_titanic, test_titanic], axis = 0, sort = False)
df_full.describe()
import pandas_profiling as pdp # pandas_profilingのインポート
pdp.ProfileReport(train_titanic) # レポートの作成
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Survived', data=train_titanic)
plt.title('死亡者と生存者の数')
plt.xticks([0, 1],['dead', 'survive'])
plt.show()
# 死亡者と生存者数を表示する
display(train_titanic['Survived'].value_counts())
# 死亡者と生存者割合を表示する
display(train_titanic['Survived'].value_counts()/len(train_titanic['Survived']))
# 男女別の生存者数を可視化
sns.countplot(x='Sex', hue='Survived', data=train_titanic)
plt.title('Number of deaths and survivors, by gender')
plt.legend(['dead', 'survived'])
plt.show()
# SexとSurvivedをクロス集計する
display(pd.crosstab(train_titanic['Sex'], train_titanic['Survived']))
# クロス集計しSexごとに正規化する
display(pd.crosstab(train_titanic['Sex'], train_titanic['Survived'], normalize='index'))


# チケットクラス別の生存者数を可視化
sns.countplot(x='Pclass', hue='Survived', data=train_titanic)
plt.title('Number of deaths and survivors by ticket class')
plt.legend(['dead', 'survived'])
plt.show()
# PclassとSurvivedをクロス集計する
display(pd.crosstab(train_titanic['Pclass'], train_titanic['Survived']))
# クロス集計しPclassごとに正規化する
display(pd.crosstab(train_titanic['Pclass'], train_titanic['Survived'], normalize='index'))
# 全体のヒストグラム
sns.distplot(train_titanic['Age'].dropna(), kde=False, bins=30, label='all')
# 死亡者のヒストグラム
sns.distplot(train_titanic[train_titanic['Survived'] == 0].Age.dropna(), kde=False, bins=30,
label='dead')
# 生存者のヒストグラム
sns.distplot(train_titanic[train_titanic['Survived'] == 1].Age.dropna(), kde=False, bins=30,
label='survived')
plt.title('Distribution of age of people on board') # タイトル
plt.legend() # 凡例を表示
plt.show()
# 年齢を８等分し、CategoricalAgeという変数を作成
train_titanic['CategoricalAge'] = pd.cut(train_titanic['Age'], 8)
# CategoricalAgeとSurvivedをクロス集計する
display(pd.crosstab(train_titanic['CategoricalAge'], train_titanic['Survived']))
# クロス集計しCategoricalAgeごとに正規化する
display(pd.crosstab(train_titanic['CategoricalAge'], train_titanic['Survived'],
normalize='index'))
sns.countplot(x='SibSp', data=train_titanic)
plt.title('Number of siblings/spouses riding in the car')
plt.show()
# SibSpが0か1であればそのまま、2以上であれば2である特徴量SibSp_0_1_2overを作成
train_titanic['SibSp_0_1_2over'] = [i if i <=1 else 2 for i in train_titanic['SibSp']]
# SibSp_0_1_2overごとに集計し、可視化
sns.countplot(x='SibSp_0_1_2over', hue='Survived', data=train_titanic)
plt.legend(['dead', 'survived'])
plt.xticks([0,1,2], ['0', '1', '2 or more'])
plt.title('Number of fatalities and survivors by number of siblings/spouses in the car')
plt.show()
# SibSpとSurvivedをクロス集計する
display(pd.crosstab(train_titanic['SibSp_0_1_2over'], train_titanic['Survived']))
# クロス集計しSibSpごとに正規化する
display(pd.crosstab(train_titanic['SibSp_0_1_2over'], train_titanic['Survived'],
normalize='index'))
sns.countplot(x = "Parch", data = train_titanic)
plt.title('Number of parents/children in the car')
plt.show()
# 2以下であればそのままの数、3以上は3という変換を行う
train_titanic['Parch_0_1_2_3over'] = [i if i <=2 else 3 for i in train_titanic['Parch']]
# Parch_0_1_2_3overごとに集計し可視化
sns.countplot(x='Parch_0_1_2_3over',hue='Survived', data=train_titanic)
plt.title('同乗している両親・子供の数別の死亡者と生存者の数')
plt.legend(['dead','survived'])
plt.xticks([0, 1, 2, 3], ['0', '1', '2', '3 or more'])
plt.xlabel('Parch')
plt.show()
# ParchとSurvivedをクロス集計する
display(pd.crosstab(train_titanic['Parch_0_1_2_3over'], train_titanic['Survived']))
# クロス集計しParchごとに正規化する
display(pd.crosstab(train_titanic['Parch_0_1_2_3over'], train_titanic['Survived'],
normalize='index'))
#SibSpとParchが同乗している家族の数。1を足すと家族の人数となる
train_titanic['FamilySize'] = train_titanic['SibSp'] + train_titanic['Parch'] + 1
# IsAloneを0とし、2行目でFamilySizeが2以上であれば1にしている
train_titanic['IsAlone'] = 0
train_titanic.loc[train_titanic['FamilySize'] >= 2, 'IsAlone'] = 1
# IsAloneごとに可視化
sns.countplot(x='IsAlone', hue='Survived', data=train_titanic)
plt.xticks([0, 1], ['1', '2 or more'])
plt.legend(['dead', 'survied'])
plt.title('Number of fatalities and survivors by vessel with more than one or two people on board')
plt.show()
# IsAloneとSurvivedをクロス集計する
display(pd.crosstab(train_titanic['IsAlone'], train_titanic['Survived']))
# クロス集計しIsAloneごとに正規化する
display(pd.crosstab(train_titanic['IsAlone'], train_titanic['Survived'], normalize='index'))
train_titanic.head()
sns.distplot(train_titanic['Fare'].dropna(), kde=False, hist=True)
plt.title('Distribution of fares')
plt.show()
train_titanic['CategoricalFare'] = pd.qcut(train_titanic['Fare'], 4)
train_titanic[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'],as_index=False).mean()
# CategoricalFareとSurvivedをクロス集計する
display(pd.crosstab(train_titanic['CategoricalFare'], train_titanic['Survived']))
# クロス集計しCategoricalFareごとに正規化する
display(pd.crosstab(train_titanic['CategoricalFare'], train_titanic['Survived'],normalize='index'))
train_titanic['Name'][0:5]
set(train_titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False))
train_titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False).value_counts()
# trainにTitle列を作成、Title列の値は敬称
train_titanic['Title'] = train_titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# trainのTitle列の値ごとに平均値を算出
train_titanic.groupby('Title').mean()['Age']
train_titanic.head()
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
# testにもtitle列を作成
test_titanic['Title'] = test_titanic.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# リスト内包表記を用いて変換
train_titanic['Title_num'] = [title_to_num(i) for i in train_titanic['Title']]
test_titanic['Title_num'] = [title_to_num(i) for i in test_titanic['Title']]
train_titanic.head()
# SexとEmbarkedのOne-Hotエンコーディング
train = pd.get_dummies(train_titanic, columns=['Sex', 'Embarked'])
test = pd.get_dummies(test_titanic, columns=['Sex', 'Embarked'])
# 不要な列の削除
train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
# trainの表示
display(train.head())
display(train)
#目的変数のどの変数が相関しているかを確認.
train.corr()
import seaborn as sns 
fig = plt.subplots(figsize=(10, 8))
sns.heatmap(train.corr(), vmax=1, annot=True,fmt='.2f');
X_train = train.drop(['Survived'], axis=1) # X_trainはtrainのSurvived列以外
y_train = train['Survived'] # y_trainはtrainのSurvived列
X_train.head()
X_train.isnull().sum()
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# X_trainとy_trainをtrainとvalidに分割
train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train,
test_size=0.33, random_state=0)
# lgb.Datasetでtrainとvalidを作っておく
lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(valid_x, valid_y)
# パラメータを定義
lgbm_params = {'objective': 'binary'}
# lgb.trainで学習
evals_result = {}
gbm = lgb.train(params=lgbm_params,
train_set=lgb_train,
valid_sets=[lgb_train, lgb_eval],
early_stopping_rounds=20,
evals_result=evals_result,
verbose_eval=10);
oof = (gbm.predict(valid_x) > 0.5).astype(int)
print('score', round(accuracy_score(valid_y, oof)*100,2))
import matplotlib.pyplot as plt
plt.plot(evals_result['training']['binary_logloss'], label='train_loss')
plt.plot(evals_result['valid_1']['binary_logloss'], label='valid_loss')
plt.legend()
test_pred = (gbm.predict(test) > 0.5).astype(int)
gender_submission['Survived'] = test_pred
gender_submission.to_csv('train_test_split.csv', index=False)
from sklearn.model_selection import KFold
# 3分割交差検証を指定し、インスタンス化
kf = KFold(n_splits=3, shuffle=True)
# スコアとモデルを格納するリスト
score_list = []
models = []
for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print(f'fold{fold_ + 1} start')
    train_x = X_train.iloc[train_index]
    valid_x = X_train.iloc[valid_index]
    train_y = y_train[train_index]
    valid_y = y_train[valid_index]
    # lab.Datasetを使って、trainとvalidを作っておく
    lgb_train= lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(valid_x, valid_y)
    # パラメータを定義
    lgbm_params = {'objective': 'binary'}
    # lgb.trainで学習
    gbm = lgb.train(params = lgbm_params,
    train_set = lgb_train,
    valid_sets= [lgb_train, lgb_valid],
    early_stopping_rounds=20,
    verbose_eval=-1 # 学習の状況を表示しない
    )

    oof = (gbm.predict(valid_x) > 0.5).astype(int)
    score_list.append(round(accuracy_score(valid_y, oof)*100,2))
    models.append(gbm) # 学習が終わったモデルをリストに入れておく
    print(f'fold{fold_ + 1} end\n')
print(score_list, '平均score', round(np.mean(score_list), 2))
# テストデータの予測を格納する、418行3列のnumpy行列を作成
test_pred = np.zeros((len(test), 3))
for fold_, gbm in enumerate(models):
    pred_ = gbm.predict(test) # testを予測
    test_pred[:, fold_] = pred_
pred = (np.mean(test_pred, axis=1) > 0.5).astype(int)
gender_submission['Survived'] = pred
gender_submission.to_csv('3-fold_cross-validation.csv',index=False)
from sklearn.model_selection import KFold
# 10分割交差検証を指定し、インスタンス化
kf = KFold(n_splits=10, shuffle=True)
# スコアとモデルを格納するリスト
score_list = []
models = []
for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print(f'fold{fold_ + 1} start')
    train_x = X_train.iloc[train_index]
    valid_x = X_train.iloc[valid_index]
    train_y = y_train[train_index]
    valid_y = y_train[valid_index]
    # lab.Datasetを使って、trainとvalidを作っておく
    lgb_train= lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(valid_x, valid_y)
    # パラメータを定義
    lgbm_params = {'objective': 'binary'}
    # lgb.trainで学習
    gbm = lgb.train(params = lgbm_params,
    train_set = lgb_train,
    valid_sets= [lgb_train, lgb_valid],
    early_stopping_rounds=20,
    verbose_eval=-1 # 学習の状況を表示しない
    )

    oof = (gbm.predict(valid_x) > 0.5).astype(int)
    score_list.append(round(accuracy_score(valid_y, oof)*100,2))
    models.append(gbm) # 学習が終わったモデルをリストに入れておく
    print(f'fold{fold_ + 1} end\n')
print(score_list, '平均score', round(np.mean(score_list), 2))
# テストデータの予測を格納する、418行3列のnumpy行列を作成
test_pred = np.zeros((len(test), 10))
for fold_, gbm in enumerate(models):
    pred_ = gbm.predict(test) # testを予測
    test_pred[:, fold_] = pred_
pred = (np.mean(test_pred, axis=1) > 0.5).astype(int)
gender_submission['Survived'] = pred
gender_submission.to_csv('10-fold_cross-validation.csv',index=False)
