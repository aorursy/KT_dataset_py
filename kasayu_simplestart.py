# データ操作のためのライブラリをインポート

import pandas as pd



# データをインポート

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# データサイズ、特徴量、データ型などを確認

# object型が含まれているので、category型に後々変換する

train.info()
# データサイズ、特徴量、データ型などを確認

# object型が含まれているので、category型に後々変換する

test.info()
# object型をcategory型に変換する関数を定義

def categorize(x):

    object_columns = x.select_dtypes(include='object').columns

    x[object_columns] = x[object_columns].astype('category')

    

    print(f'complited data categorization')

    



# object型をcategory型に変換

categorize(train)

categorize(test)
# データを説明変数と目的変数に分ける

train_x = train.drop(columns='SalePrice')

train_y = train['SalePrice']



# データを学習用(tr_x, tr_y)と評価用(val_x, val_y)に分ける

from sklearn.model_selection import train_test_split

tr_x, val_x = train_test_split(train_x, test_size=0.3)

tr_y, val_y = train_test_split(train_y, test_size=0.3)
# モデルのパラメータを設定

random_seed = 12345

params = {

    # 予測手法として回帰を選択

    'objective': 'regression',

    # lossはlogloss

    'metric': 'rmse',

    # 学習率（デフォルト0.1）

    'learning_rate': 0.05,

    # 木の深さ、-1だと制限無し（デフォルト-1）

    'max_depth': -1,

    # 乱数を固定

    'random_seed': random_seed,

    # 学習する回数（デフォルト100）

    'num_boost_round': 1000,

    # 精度が上がらなくなったら学習を打ち切る

    'early_stopping_rounds': 50,

    # 学習途中を表示する間隔

    'verbose_eval': 10

}
# 機械学習フレームワークをインポート

import lightgbm as lgb



# データをモデルにセット

train_data = lgb.Dataset(tr_x, label=tr_y)

val_data = lgb.Dataset(val_x, label=val_y, reference=train_data)



# trainデータで学習

gbm_reg = lgb.train(params, train_data, valid_sets=val_data)
# 評価用データで予測

val_pred = gbm_reg.predict(val_x, num_iteration=gbm_reg.best_iteration)



# 比較のため学習用データで予測

tr_pred = gbm_reg.predict(tr_x, num_iteration=gbm_reg.best_iteration)



# RMSEを計算

import numpy as np

from sklearn.metrics import mean_squared_error



def calc_rmse(x, y):

    mse = mean_squared_error(x, y)

    rmse = np.sqrt(mse)

    

    return rmse



# trainデータとvalidationデータで予測

print (f'Training RMSE:{calc_rmse(tr_y, tr_pred)}',f'Validation RMSE:{calc_rmse(val_y, val_pred)}')
# テストデータで予測

test_pred = gbm_reg.predict(test)

print(f'Test Score:{gbm_reg.predict(test)}')
# 提出用のテーブルを作成

submission = pd.DataFrame({

    'Id': test['Id'],

    'SalePrice': test_pred

})



# 提出用csvファイル作成

submission.to_csv('submission.csv', index=False)