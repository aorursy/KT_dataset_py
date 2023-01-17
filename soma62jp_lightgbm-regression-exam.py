# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy import stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#EDA

# 求めたいSalePriceの基本統計量を調べる

(mu, sigma) = stats.norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

print( '\n skew = {:.2f}\n'.format(stats.skew(train['SalePrice'])))
# EDAのの2

# データの素性を調べる

train.info()
# 欠損値を調べる

# ただしlightGBMは欠損値の考慮が不要

np.set_printoptions(threshold=np.inf)        # 全件表示設定

pd.set_option('display.max_rows',10000)      # 1000件表示設定



train.isnull().sum()
# 基本統計量

train.describe()
# trainデータとtestデータに分ける

#数値データ一括log変換

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



# One-Hotエンコーディング

all_data = pd.get_dummies(all_data)                      



X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train['SalePrice']
import lightgbm as lgb

import optuna

from sklearn.model_selection import train_test_split,KFold,cross_validate

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
# 学習データ = 75% テストデータ = 25%　に分割

train_x, test_x, train_y, test_y = train_test_split(X_train, y, test_size=0.25, 

                                                    shuffle = True , random_state = 0)
# LightGBM用のDatasetに格納

dtrain = lgb.Dataset(train_x, label=train_y)

dtest = lgb.Dataset(test_x, label=test_y)
#------------------------LightGBM Model 最適化-----------------------

# ハイパーパラメータ検索ライブラリ「Optuna」を使用

def objectives(trial):

    

    # --optunaでのハイパーパラメータサーチ範囲の設定

    # 回帰問題

    # rmse最適化

    # 勾配ブースティングを使用する

    params = {

        #fixed

        'boost_from_average': True, ## ONLY NEED FOR LGB VERSION 2.1.2

        "objective": "regression",

        'boosting_type':'gbdt',

        'max_depth':-1,

        'learning_rate':0.1,

        'n_estimators': 1000,

        'metric':'rmse',



        #variable

        'num_leaves': trial.suggest_int('num_leaves', 10, 300),

        'reg_alpha': trial.suggest_loguniform('reg_alpha',0.001, 10),

        'reg_lambda':trial.suggest_loguniform('reg_lambda', 0.001, 10),

    }



    # LightGBMで学習+予測

    model = lgb.LGBMRegressor(**params,random_state=0)

    

    # kFold交差検定で決定係数を算出し、各セットの平均値を返す

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    scores = cross_validate(model, X=train_x, y=train_y,scoring='r2',cv=kf)   



    # 最小化問題とするので1.0から引く

    return 1.0 - scores['test_score'].mean()





# optunaによる最適化呼び出し

opt = optuna.create_study(direction='minimize')

opt.optimize(objectives, n_trials=20)
# 実行結果表示

print('最終トライアル回数:{}'.format(len(opt.trials)))

print('ベストトライアル:')

trial = opt.best_trial

print('値:{}'.format(trial.value))

print('パラメータ:')

for key, value in trial.params.items():

    print('{}:{}'.format(key, value))
from sklearn.metrics import mean_squared_error,r2_score



gbm_best = lgb.train(trial.params, dtrain)



def model_Eval(testX,testY):

    

    predict_best = gbm_best.predict(testX)

    print(predict_best.shape)



    np.set_printoptions(threshold=10)        # 10件表示設定

    pd.set_option('display.max_rows',10)      # 10件表示設定



    preds = pd.DataFrame({"preds":predict_best, "true":testY})

    display(preds.head(10))

    

    # 重要度プロット

    lgb.plot_importance(gbm_best,importance_type='split',max_num_features = 20,figsize=(12,6))



    preds.dropna(axis=0)



    # 残差プロット

    preds["residuals"] = preds["true"] - preds["preds"]

    preds.plot(x = "preds", y = "residuals",kind = "scatter")



    # モデルのあてはめ

    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)



    ax.scatter(preds["true"], preds["preds"],label="LigntGBM Model Fitting")

    ax.set_xlabel('predicted')

    ax.set_ylabel('true')

    ax.set_aspect('equal')



    # mseとr2を求める

    mse = mean_squared_error(preds["true"], preds["preds"])

    print('mse:',mse)

    r2 = r2_score(preds["true"], preds["preds"])

    print('r2:',r2)
# モデルデータの検証

model_Eval(train_x,train_y)
# 過学習の確認

# テストデータの検証

model_Eval(test_x,test_y)
# 予測開始

y_pred_lgb = gbm_best.predict(X_test)
pred_df = pd.DataFrame(y_pred_lgb, index=test["Id"], columns=["SalePrice"])

pred_df
pred_df.to_csv('output.csv', header=True, index_label='Id')