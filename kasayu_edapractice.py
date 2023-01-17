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
# 学習用テーブルデータを確認
train.head(10)
# テスト用テーブルデータを確認
test.head(10)
null_info_df = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)).head(20).rename(columns={0:'null_count'})
null_info_df['null_percent(%)'] = null_info_df['null_count']/len(train) * 100
null_info_df
# flag_is_null = train.isnull().any()
# train[(flag_is_null==True).index].isnull().sum()
# flag_is_null
# 統計量の確認（データ数、平均値、標準偏差、最小・最大値、25/50/75%の分位点）
train['SalePrice'].describe()
# ヒストグラムの確認（seaborn：グラフ描画ライブラリ）
import seaborn as sns
sns.distplot(train['SalePrice'])
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
# 描画用のライブラリをインポート
import matplotlib.pyplot as plt

# ヒートマップを作成
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
import numpy as np

k = 10 # ヒートマップの特徴変数の数
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index # 
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# ユニーク値のIDを削除
# 提出の際に使うので、別の値として保管しておく
test_id = test['Id']
train = train.drop(columns='Id')
test = test.drop(columns='Id')
train.head()
train = train.drop(columns='GarageArea')
train = train.drop(columns='TotRmsAbvGrd')
test = test.drop(columns='GarageArea')
test = test.drop(columns='TotRmsAbvGrd')
# import numpy as np
# train['SalePrice'] = np.log(train['SalePrice'])
# sns.distplot(train['SalePrice'])
# print("Skewness: %f" % train['SalePrice'].skew())
# print("Kurtosis: %f" % train['SalePrice'].kurt())
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
# from sklearn.model_selection import KFold
# import lightgbm as lgb

# scores_rmse=[]

# # 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
# kf = KFold(n_splits=4, shuffle=True, random_state=random_seed)
# for tr_idx, val_idx in kf.split(train_x):
    
#     # 学習データを学習データとバリデーションデータに分ける
#     tr_x, val_x = train_x.iloc[tr_idx], train_x.iloc[val_idx]
#     tr_y, val_y = train_y.iloc[tr_idx], train_y.iloc[val_idx]

#     # データをモデルにセット
#     train_data = lgb.Dataset(tr_x, label=tr_y)
#     eval_data = lgb.Dataset(val_x, label=val_y, reference=train_data)
   
#     # モデルの学習を行う
#     gbm_reg = lgb.train(params, 
#                         train_data, 
#                         valid_sets=eval_data
#                        )
    
#     # バリデーションデータで予測値を求める
#     validation_pred = gbm_reg.predict(val_x, num_iteration=gbm_reg.best_iteration)

#     # 各foldでの評価指標の値を保存
#     scores_rmse.append(validation_pred)

# # 各foldのスコアの平均を出力する
# rmse_mean = np.mean(scores_rmse)
# print(f'rmse_mean: {rmse_mean:.4f}')
# # trainデータとvalidationデータで予測
# print (f'Training Score:{gbm_reg.predict(tr_x)}',f'Validation Score:{gbm_reg.predict(val_x)}')
# テストデータで予測
test_pred = gbm_reg.predict(test)
print(f'Test Score:{gbm_reg.predict(test)}')
# 提出用のテーブルを作成
submission = pd.DataFrame({
    'Id': test_id,
    'SalePrice': test_pred
})

# 提出用csvファイル作成
submission.to_csv('submission.csv', index=False)