# ライブラリのインポート

import pandas as pd # 基本ライブラリ

import numpy as np # 基本ライブラリ

import matplotlib.pyplot as plt # グラフ描画用

import seaborn as sns; sns.set() # グラフ描画用

import warnings # 実行に関係ない警告を無視

warnings.filterwarnings('ignore')

import lightgbm as lgb #LightGBM

from sklearn import datasets

from sklearn.model_selection import train_test_split # データセット分割用

from sklearn.metrics import accuracy_score # モデル評価用(正答率)

from sklearn.metrics import log_loss # モデル評価用(logloss)     

from sklearn.metrics import roc_auc_score # モデル評価用(auc)

from sklearn import preprocessing



# データフレームを綺麗に出力する関数

import IPython

def display(*dfs, head=True):

    for df in dfs:

        IPython.display.display(df.head() if head else df)
# データセットの読み込み

train = pd.read_csv("../input/otto-group-product-classification-challenge/train.csv")

test = pd.read_csv("../input/otto-group-product-classification-challenge/test.csv")

sample = pd.read_csv("../input/otto-group-product-classification-challenge/sampleSubmission.csv")



# 説明変数,目的変数にデータを分類

tage = train.target.values # 目的変数(target)

X = train.drop("id",axis = 1)# 説明変数(target以外の特徴量)

X = X.drop("target",axis = 1)# 説明変数(target以外の特徴量)

y = test.drop("id",axis = 1) # IDは不要なので除去



lbl_enc = preprocessing.LabelEncoder()

label_encoded_y = lbl_enc.fit_transform(tage)#目的変数を数値に変換

X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y,test_size=0.20, random_state=29)



train_data = lgb.Dataset(X_train, label=y_train)

eval_data = lgb.Dataset(X_test, label=y_test, reference= train_data)



params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'multiclass',

    'num_class': 9,

    'verbose': 2,

}

model = lgb.train(

    params,

    train_data,

    valid_sets=eval_data,

    num_boost_round=100,

    verbose_eval=10,

)
y.head()
print(label_encoded_y)
# テストデータのクラス予測確率 (各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す)

y_pred_prob = model.predict(y, num_iteration=model.best_iteration)

y_pred_prob = pd.DataFrame(y_pred_prob, index=sample.id.values, columns=sample.columns[1:])

y.to_csv("tes.csv",index=False)

y_pred_prob.to_csv("out1.csv", index_label='id')