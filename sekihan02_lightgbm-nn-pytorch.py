import numpy as np

import pandas as pd



# データの読み込み

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

sub = pd.read_csv('../input/titanic/gender_submission.csv')    # サンプルの予測データ



# データの確認

print(train.shape)

train.head()
print(test.shape)

test.head()
# データ形式の確認

train.info()
import pandas_profiling



train.profile_report()
# 説明変数の確認

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



def plot_count(feature, title, df, size=1):

    """クラス/特徴量をプロットする

    Pram:

        feature : 分析するカラム

        title : グラフタイトル

        df : プロットするデータフレーム

        size : デフォルト 1.

    """

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    # 最大20カラムをヒストグラムで表示

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        # サイズ2以上の時、行名を90°回転し、表示

        plt.xticks(rotation=90, size=8)

    # データ比率の表示

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()  
# Survivedの比率を確認する

plot_count(feature='Survived', title='survived', df=train, size=1)
# 円グラフで可視化

plt.pie(

    train['Survived'].value_counts(),    # データの出現頻度

    labels=train['Survived'].value_counts().index,    # ラベル名の指定

    counterclock=False,    # データを時計回りに入れる

    startangle=90,          # データの開始位置 90の場合は円の上から開始

    autopct='%1.1f%%',      # グラフ内に構成割合のラベルを小数点1桁まで表示

    pctdistance=0.8         # ラベルの表示位置

)



plt.show()
def plot_count2(feature, hue, title, df, size=1):

    """クラス/特徴量をプロットする

    Pram:

        feature : 分析するカラム

        title : グラフタイトル

        hue : 各軸をさらに分割して集計する列名

        df : プロットするデータフレーム

        size : デフォルト 1.

    """

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    # 最大20カラムをヒストグラムで表示

    g = sns.countplot(df[feature], hue=df[hue], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 3):

        # サイズ2以上の時、行名を90°回転し、表示

        plt.xticks(rotation=90, size=8)

    # 表示データ全体の割合の表示

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()  
# Ageと目的変数の関係

print('data count\n', train['Age'].value_counts().sort_values(ascending=False).index[:10])    # データの出現数を上位10個表紙

print('Null count ', train['Age'].isnull().sum())



sns.countplot(data=train, y='Age', hue='Survived', order = train['Age'].value_counts().index[:], palette='Set3')

plt.legend(loc='upper right', title='Survived')

plt.tight_layout()
# Ageと目的変数の関係を上位15データで表示

sns.countplot(data=train, y='Age', hue='Survived', order = train['Age'].value_counts().index[:15], palette='Set3')

plt.legend(loc='upper right', title='Survived')

plt.tight_layout()
# 上記のグラフが見づらいので別な表示方法でAgeと目的変数の関係（全体を表示）

plt.hist(train.loc[train['Survived'] == 0, 'Age'].dropna(), bins=30, alpha=0.5, label='0')

plt.hist(train.loc[train['Survived'] == 1, 'Age'].dropna(), bins=30, alpha=0.5, label='1')

plt.xlabel('Age')

plt.ylabel('count')

plt.legend(title='Survived')

plt.show()
# SibSp配偶者の数と目的変数との関係=

print('data count\n', train['SibSp'].value_counts())

print('Null count ', train['SibSp'].isnull().sum())

sns.countplot(data=train, x='SibSp', hue='Survived', palette='Set3')
plot_count2(df=train, feature='SibSp', hue='Survived', title='SibSp', size=2.5)
# Parchと目的変数との関係

print('data count\n', train['Parch'].value_counts())

print('Null count ', train['Parch'].isnull().sum())



plot_count2(df=train, feature='SibSp', hue='Survived', title='SibSp', size=2.5)

# sns.countplot(data=train, x='Parch', hue='Survived', palette='Set3')
# Fareと目的変数との関係

print('data count\n', train['Fare'].value_counts().sort_values(ascending=False).index[:10])

print('Null count ', train['Fare'].isnull().sum())



plot_count2(df=train, feature='Fare', hue='Survived', title='Fare', size=3.5)
sns.countplot(data=train, x='Fare', hue='Survived', order=train['Fare'].value_counts().sort_values(ascending=False).index[:10], palette='Set3')
# Fareと目的変数との関係

plt.hist(train.loc[train['Survived'] == 0, 'Fare'].dropna(), range=(0, 250), bins=25, alpha=0.5, label='0') 

plt.hist(train.loc[train['Survived'] == 1, 'Fare'].dropna(), range=(0, 250), bins=25, alpha=0.5, label='1') 

plt.xlabel('Fare')

plt.ylabel('count')

plt.legend(title='Survived')

plt.xlim(-5, 250)
# Pclassチケットクラスと目的変数との関係

print('data count\n', train['Pclass'].value_counts())

print('Null count ', train['Pclass'].isnull().sum())



plot_count2(df=train, feature='Pclass', hue='Survived', title='Pclass', size=2)
# Sexと目的変数との関係

print('data count\n', train['Sex'].value_counts())

print('Null count ', train['Sex'].isnull().sum())



plot_count2(df=train, feature='Sex', hue='Survived', title='sSex', size=2)

# sns.countplot(x='Sex', hue='Survived', data=train)
# Embarked出港地と目的変数との関係

print('data count\n', train['Embarked'].value_counts())

print('Null count ', train['Embarked'].isnull().sum())



plot_count2(df=train, feature='Embarked', hue='Survived', title='Embarked', size=2)
# ParchとSibSpを足し合わせてFamilySizeを新しく作成

train['FamilySize'] = train['Parch'] + train['SibSp'] + 1

test['FamilySize'] = test['Parch'] + test['SibSp'] + 1



# 表示確認の可視化

plot_count2(df=train, feature='FamilySize', hue='Survived', title='FamilySize', size=3)
# 家族数1の特徴量IsAloneを作成

train['IsAlone'] = 0

train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1    # 行がtrain['FamilySize'] == 1のとき'IsaAlone'を1に



test['IsAlone'] = 0

test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
# データの縦連結

data = pd.concat([train, test], sort=False)

data.head()
# 欠損値の確認

data.isnull().sum()
# データ形式の確認

data.info()
# 年齢を中央値で補完

data['Age'].fillna(data['Age'].median(), inplace=True)
# 運賃を平均で補完

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
# 頻度（出現回数）をカウント

print(data['Embarked'].value_counts())



# 欠損値はSとして補完

data['Embarked'].fillna(('S'), inplace=True)

# S=0 C=1 Q=2 にint型で変換

data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)



# 変換後の型の確認

data['Embarked'].dtype
# 性別を'male'=0, 'female'=1で変換

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
# 上記のラベルエンコーディングをまとめて実行するなら

# from sklearn.preprocessing import LabelEncoder



# # それぞれのカテゴリ変数にlabel encodingを適用する

# for c in ['Sex', 'Embarked']:

#     # 学習データに基づいてどう変換するかを定める

#     le = LabelEncoder()

#     le.fit(data[c].fillna('NA'))



#     # 学習データ、テストデータを変換する

#     data[c] = le.transform(data[c].fillna('NA'))
data.isnull().sum()
# 学習に使用しないカラムリストの作成

del_colum = ['PassengerId', 'Name', 'Ticket', 'Cabin']

data.drop(del_colum, axis=1, inplace=True)

# 結合していたデータを再度訓練データとテストデータに分割

train = data[:len(train)]

test = data[len(train):]



# 目的変数と説明変数に分割

y_train = train['Survived']    # 目的変数

X_train = train.drop('Survived', axis=1)    # 訓練データの説明変数

X_test = test.drop('Survived', axis=1)    # テストデータの説明変数
print(X_train.shape, y_train.shape, X_test.shape)
X_train.info()
# 学習用データを学習用・検証用に分割する

from sklearn.model_selection import train_test_split



# train:valid = 7:3

X_train, X_valid, y_train, y_valid = train_test_split(

    X_train,             # 対象データ1

    y_train,             # 対象データ2

    test_size=0.3,       # 検証用データを3に指定

    stratify=y_train,    # 訓練データで層化抽出

    random_state=42

)
# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)

categorical_features = ['Embarked', 'Pclass', 'Sex']
# LightGBMで学習の実施

import lightgbm as lgb

# データセットの初期化

lgb_train = lgb.Dataset(

    X_train,

    y_train,

    categorical_feature=categorical_features

)



lgb_valid = lgb.Dataset(

    X_valid,

    y_valid,

    reference=lgb_train,    # 検証用データで参照として使用する訓練データの指定

    categorical_feature=categorical_features

)



# パラメータの設定

params = {

    'objective':'binary'    # logistic –バイナリ分類のロジスティック回帰 (多値分類ならmultiでsoftmax)

}



lgb_model = lgb.train(

    params,    # パラメータ

    lgb_train,    # 学習用データ

    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ

    verbose_eval=10,    # 検証データは10個

    num_boost_round=1000,    # 学習の実行回数の最大値

    early_stopping_rounds=10    # 連続10回学習で検証データの性能が改善しない場合学習を打ち切る

)
# 特徴量重要度の算出 (データフレームで取得)

cols = list(X_train.columns) # 特徴量名のリスト(目的変数CRIM以外)

f_importance = np.array(lgb_model.feature_importance()) # 特徴量重要度の算出

f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)

df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})

df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート

display(df_importance)
# 特徴量重要度の可視化

n_features = len(df_importance) # 特徴量数(説明変数の個数) 

df_plot = df_importance.sort_values('importance') # df_importanceをプロット用に特徴量重要度を昇順ソート 

f_imoprtance_plot = df_plot['importance'].values # 特徴量重要度の取得 

plt.barh(range(n_features), f_imoprtance_plot, align='center') 

cols_plot = df_plot['feature'].values # 特徴量の取得 

plt.yticks(np.arange(n_features), cols_plot)  # x軸,y軸の値の設定

plt.xlabel('Feature importance') # x軸のタイトル

plt.ylabel('Feature') # y軸のタイトル
# 推論                 

lgb_y_pred = lgb_model.predict(

    X_test,    # 予測を行うデータ

    num_iteration=lgb_model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。

)

# 結果の表示

lgb_y_pred[:10]
# 予測結果の0.5を閾値として2値分類

lgb_y_pred = (lgb_y_pred > 0.5).astype(int)

# 結果の表示

lgb_y_pred[:10]
# 予測データをcsvに変換

sub = pd.read_csv('../input/titanic/gender_submission.csv')    # サンプルの予測データ

sub['Survived'] = lgb_y_pred



sub.to_csv('submit_lightgbm.csv', index=False)

sub.head()
import optuna

from sklearn.metrics import log_loss    # 評価指標としてcross entropyを使用します（予測と正解の確率分布の誤差を確認）



# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)

categorical_features = ['Embarked', 'Pclass', 'Sex']



# 学習内容の定義

def objective(trial):

    # パラメータの設定

    param = {

        'objective': 'binary',    # logistic –バイナリ分類のロジスティック回帰 (多値分類ならmultiでsoftmax)

        'metric': 'binary_logloss',

        'max_bin':trial.suggest_int('max_bin', 100, 500),

        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),

        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),

        'num_leaves': trial.suggest_int('num_leaves', 2, 256),

        'learning_rate':0.01,

        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),

        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),

        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),

        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        'num_leaves': trial.suggest_int('num_leaves', 32, 128)

    }

    # データセットの初期化

    lgb_train = lgb.Dataset(

        X_train,

        y_train,

        categorical_feature=categorical_features

    )



    lgb_valid = lgb.Dataset(

        X_valid,

        y_valid,

        reference=lgb_train,    # 検証用データで参照として使用する訓練データの指定

        categorical_feature=categorical_features

    )



    model = lgb.train(

        params,    # パラメータ

        lgb_train,    # 学習用データ

        valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ

        verbose_eval=10,    # 検証データは10個

        num_boost_round=1000,    # 学習の実行回数の最大値

        early_stopping_rounds=10    # 連続10回学習で検証データの性能が改善しない場合学習を打ち切る

    )



    # 推論                 

    y_pred = model.predict(

        X_valid,    # 予測を行うデータ

        num_iteration=model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。

    )

    # 評価

    score = log_loss(

        y_valid,    # 正解値

        y_pred      # 予測結果

    )

    return score
# ハイパーパラメーターチューニングの実行

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))

study.optimize(objective, n_trials=40)
# ベストパラメーターの表示

study.best_params
# 訓練用と検証用のデータの割合をできるだけそろえるように分割するライブラリ

from sklearn.model_selection import StratifiedKFold



y_preds = []    # 検証結果の格納先

models = []    # モデルのパラメータの格納先

oof_train = np.zeros((len(X_train), ))    # 学習で使用されなかったデータ

# 5分割して交差検証

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)

categorical_features = ['Embarked', 'Pclass', 'Sex']



# パラメータの設定

param = {

    'objective': 'binary',    # logistic –バイナリ分類のロジスティック回帰 (多値分類ならmultiでsoftmax)

    'metric': 'binary_logloss',

    'max_bin':study.best_params['max_bin'],

    'lambda_l1': study.best_params['lambda_l1'],

    'lambda_l2': study.best_params['lambda_l2'],

    'num_leaves': study.best_params['num_leaves'],

    'learning_rate':0.01,

    'feature_fraction': study.best_params['feature_fraction'],

    'bagging_fraction': study.best_params['bagging_fraction'],

    'bagging_freq': study.best_params['bagging_freq'],

    'min_child_samples': study.best_params['min_child_samples'],

    'num_leaves': study.best_params['num_leaves'],

}



for train_index, valid_index in cv.split(X_train, y_train):

    # 訓練データを訓練データとバリデーションデータに分ける

    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]    # 分割後の訓練データ

    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]    # 分割後の検証データ

    

    # データセットの初期化

    lgb_train = lgb.Dataset(

        X_tr,

        y_tr,

        categorical_feature=categorical_features

    )

    lgb_eval = lgb.Dataset(

        X_val,

        y_val,

        reference=lgb_train,    # 検証用データで参照として使用する訓練データの指定

        categorical_feature=categorical_features

    )

    

    lgb_model = lgb.train(

        params,    # パラメータ

        lgb_train,    # 学習用データ

        valid_sets=[lgb_train, lgb_eval],    # 訓練中に評価されるデータ

        verbose_eval=10,    # 検証データは10個

        num_boost_round=1000,    # 学習の実行回数の最大値

        early_stopping_rounds=10    # 連続10回学習で検証データの性能が改善しない場合学習を打ち切る

    )

    # 検証の実施

    oof_train[valid_index] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

    # 予測の実施

    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    

    y_preds.append(y_pred)

    models.append(lgb_model)
# 検証データをCSVファイルとして保存

pd.DataFrame(oof_train).to_csv('oof_train_skfold.csv', index=False)

print(oof_train[:10])    # 検証結果の表示

scores = [

    m.best_score['valid_1']['binary_logloss'] for m in models

]

score = sum(scores) / len(scores)



print('===CV scores===')

# 交差検証ごとの結果

print(scores)

# 交差検証の結果の平均

print(score)
# 正解率の表示

from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train > 0.5).astype(int)



print("LightGBMの正解率 : {:.2f}".format(accuracy_score(y_train, y_pred_oof)))
y_sub = sum(y_preds) / len(y_preds)

y_sub_lgb = (y_sub > 0.5).astype(int)

y_sub_lgb[:10]
# 予測データをcsvに変換

sub_lgb = pd.DataFrame({'PassengerId': sub['PassengerId'], 'Survived': y_sub_lgb})

sub_lgb.to_csv('submission_lightgbm_skfold.csv', index=False)



sub_lgb.head()
import random

import torch



def seed_everything(seed=1234):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



seed_everything(seed=42)
# デバイスモードの取得

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)    # デバイスモードの確認
print(X_train.isnull().sum(), y_train.isnull().sum(), X_test.isnull().sum())
# データをNumpy配列に変換

X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

# データをテンソルに変換

X_train = torch.from_numpy(X_train).float().to(device)

y_train = torch.from_numpy(y_train).long().to(device)

X_test = torch.from_numpy(X_test).float().to(device)
import torch.nn as nn



# 定数

INPUT_SIZE = X_train.shape[1]    # 入力層のニューロン数

HIDDEN_SIZE = 512     # 隠れ層のニューロン数

OUTPUT_CLASSES = 2      # 出力層のニューロン数

DROPOUT_PROBABILITY = 0.5    # ドロップアウト確率



# モデルの定義

class NeuralNetwork(nn.Module):

    def __init__(self):

        # 継承したnn.Module親クラスを初期化

        super(NeuralNetwork, self).__init__()

        # 層の定義

        self.fc1 = nn.Linear(

            INPUT_SIZE,    # 入力層のユニット数

            HIDDEN_SIZE    # 次の層への出力ユニット数

        )

        self.fc2 = nn.Linear(

            HIDDEN_SIZE,   # 隠れ層のユニット数

            HIDDEN_SIZE    # 次の層への出力ユニット数

        )

        self.fc3 = nn.Linear(

            HIDDEN_SIZE,   # 隠れ層のユニット数

            HIDDEN_SIZE    # 次の層への出力ユニット数

        )

        self.fc4 = nn.Linear(

            HIDDEN_SIZE,   # 隠れ層のユニット数

            HIDDEN_SIZE    # 次の層への出力ユニット数

        )

        self.fc5 = nn.Linear(

            HIDDEN_SIZE,   # 隠れ層のユニット数

            OUTPUT_CLASSES # 出力層のニューロン数

        )

        # 活性化関数

        self.relu = nn.ReLU()

        # ドロップアウト層

        self.dropout = nn.Dropout(

            DROPOUT_PROBABILITY  # ドロップアウト層の確率

        )

    def forward(self, x):

        # print('Input_size : ', x.size())    # 出力サイズの確認  debag

        x = self.fc1(x)

        # print('fc1_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.dropout(x)

        # print('fc1_activation_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.fc2(x)

        # print('fc2_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.relu(x)

        # print('fc2_activation_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.fc3(x)

        # print('fc3_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.dropout(x)

        # print('fc3_activation_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.fc4(x)

        # print('fc4_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.relu(x)

        # print('fc4_activation_output_size : ', x.size())    # 出力サイズの確認  debag

        x = self.fc4(x)

        # print('Output_size : ', x.size())    # 出力サイズの確認  debag

        return x



# モデルのインスタンス化

net = NeuralNetwork().to(device)

print(net)                      # モデルの概要を出力
import torch.optim as optim    # 最適化モジュールのインポート



# 定数

LEARN_RATE = 0.01        # 学習率

# 変数

criterion = nn.CrossEntropyLoss()   # 損失関数：交差エントロピー 学習データの正解率を出力

optimizer = optim.SGD(

        net.parameters(),   # 最適化で更新する重みやバイアスのパラメータ

        lr=LEARN_RATE,        # 学習率

)
import time

from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset    # データ関連のユーティリティクラスのインポート

from torch.autograd import Variable



def init_parameters(layer):

    """パラメータ（重みとバイアス）の初期化

    引数の層が全結合層の時パラメータを初期化する

    

    Param:

      layer: 層情報

    """

    if type(layer) == nn.Linear:

        nn.init.xavier_uniform_(layer.weight)    # 重みを「一様分布のランダム値」で初期化

        layer.bias.data.fill_(0.0)               # バイアスを「0」で初期化





net.apply(init_parameters)        # 学習の前にパラメーター初期化



# 定数

start = time.time()             # 実行開始時間の取得

EPOCHS = 1000        # エポック数

BATCH_SIZE = 64        # バッチサイズ



batch_no = len(X_train) // BATCH_SIZE



loss_list = []

acc_list = []



train_loss_min = np.Inf

for epoch in range(EPOCHS):

    # 学習中の損失を格納する変数

    train_loss = 0

    # 学習中の正解数を格納する変数

    correct = 0

    total = 0           # 1ミニバッチ数を格納する変数

    for i in range(batch_no):

        # バッチの開始サイズ

        start = i*BATCH_SIZE

        # バッチの終了サイズ

        end = start+BATCH_SIZE

        # 検証データの取得

        x_var = Variable(torch.FloatTensor(X_train[start:end]))

        y_var = Variable(torch.LongTensor(y_train[start:end]))

        

        # フォワードプロパゲーションで出力結果を取得

        output = net(x_var)

        # 出力結果と正解ラベルから損失を計算し、勾配を計算

        optimizer.zero_grad()    # 勾配を0で初期化

        

        loss = criterion(output, y_var)   # 誤差（出力結果と正解ラベルの差）から損失を取得



        loss.backward()                   # 逆伝播の処理として勾配を計算（自動微分）

        optimizer.step()                  # 最適化の実施

        

        values, labels = torch.max(output, 1)  # 予測した確率の最大値を予測結果として出力

        

        correct += (labels == y_train[start:end]).sum().item()  # 正解数を取得

        total += len(y_train[start:end])              # 1ミニバッチ数の取得



        train_loss += loss.item()*BATCH_SIZE

    

    train_acc = float(correct) / len(X_train)    # 正解率を取得

    # 損失の算出ミニバッチ数分の損失の合計をミニバッチ数で割る

    train_loss = train_loss / len(X_train)

    

    # 損失値を更新したときモデルパラメーターを保存

    if train_loss <= train_loss_min:

        # print("Validation loss decreased ({:6f} ===> {:6f}). Saving the model...".format(train_loss_min,train_loss))

        torch.save(net.state_dict(), 'taitanic_model.pt')

        train_loss_min = train_loss

    

    if epoch % 100 == 0:

        # 損失や正解率などの情報を表示

        print(f'[Epoch {epoch+1:3d}/{EPOCHS:3d}]' \

              f' loss: {train_loss:.5f}, acc: {train_acc:.5f}')

    

    # logging

    loss_list.append(train_loss)

    acc_list.append(train_acc)



print('Finished Training')

# 学習終了後、学習に要した時間を出力

print("Computation time:{0:.3f} sec".format(time.time() - start))



# 損失や正解率などの情報を表示

print(f'[Epoch {epoch+1:3d}/{EPOCHS:3d}]' \

      f' loss: {train_loss:.5f}, acc: {train_acc:.5f}')
import matplotlib.pyplot as plt

%matplotlib inline



# plot learning curve

plt.figure()

plt.plot(range(EPOCHS), loss_list, 'r-', label='train_loss')

plt.legend(loc='best')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.grid()



plt.figure()

plt.plot(range(EPOCHS), acc_list, 'b-', label='train_acc')

plt.legend(loc='best')

plt.xlabel('epoch')

plt.ylabel('acc')

# plt.ylim(0, 1)

plt.grid()
X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=False) 

with torch.no_grad():

    test_result = net(X_test_var)

values, labels = torch.max(test_result, 1)

y_sub_nn= labels.data.numpy()

y_sub_nn[:5]
# 予測データをcsvに変換

sub_nn = pd.DataFrame({'PassengerId': sub['PassengerId'], 'Survived': y_sub_nn})

sub_nn.to_csv('submission_nn.csv', index=False)



sub_nn.head()
from sklearn.ensemble import RandomForestClassifier

# ジニ不純度を指標とするランダムフォレストのインスタンスを生成

forest = RandomForestClassifier(

    criterion='gini',

    n_estimators=25,

    random_state=42,

    n_jobs=2

)



# 結合していたデータを再度訓練データとテストデータに分割

train = data[:len(train)]

test = data[len(train):]



# 目的変数と説明変数に分割

y_train = train['Survived']    # 目的変数

X_train = train.drop('Survived', axis=1)    # 訓練データの説明変数

X_test = test.drop('Survived', axis=1)    # テストデータの説明変数



# 学習の実施

forest_model = forest.fit(X_train, y_train)
X_train.columns
# 特徴量重要度の算出 (データフレームで取得)

cols = list(X_train.columns) # 特徴量名のリスト(目的変数CRIM以外)

f_importance = np.array(forest_model.feature_importances_) # 特徴量重要度の算出

f_importance = f_importance / np.sum(f_importance) # 正規化(必要ない場合はコメントアウト)

df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})

df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート

display(df_importance)
# 特徴量重要度の可視化

n_features = len(df_importance) # 特徴量数(説明変数の個数) 

df_plot = df_importance.sort_values('importance') # df_importanceをプロット用に特徴量重要度を昇順ソート 

f_imoprtance_plot = df_plot['importance'].values # 特徴量重要度の取得 

plt.barh(range(n_features), f_imoprtance_plot, align='center') 

cols_plot = df_plot['feature'].values # 特徴量の取得 

plt.yticks(np.arange(n_features), cols_plot)  # x軸,y軸の値の設定

plt.xlabel('Feature importance') # x軸のタイトル

plt.ylabel('Feature') # y軸のタイトル
print("ランダムフォレストの正解率 : {:.2f}".format(forest_model.score(X_train, y_train)))
# 推論

y_sub_forest = forest_model.predict(X_test)

y_sub_forest[:10]
# int型に結果を変換

y_sub_forest = (y_sub_forest > 0.5).astype(int)

y_sub_forest[:10]
# 予測データをcsvに変換

sub_forest = pd.DataFrame({'PassengerId': sub['PassengerId'], 'Survived': y_sub_forest})

sub_forest.to_csv('submission_forest.csv', index=False)



sub_forest.head()
# LightGBMとNNの予測結果を1つのデータフレームにする

df = pd.DataFrame({'sub_lgbm': sub_lgb['Survived'].values,

                   'sub_forest': sub_forest['Survived'].values,

                   'sub_nn': sub_nn['Survived'].values})

df.head()
sub = pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived'] = df['sub_lgbm'] + (0.8 * df['sub_forest']) + (1.2 * df['sub_nn'])

sub.head()
sub['Survived'] = (sub['Survived'] >= 2).astype(int)

sub.to_csv('submission_ensemble.csv', index=False)

sub.head()