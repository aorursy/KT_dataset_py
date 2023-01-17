# データ操作

import numpy as np

import pandas as pd

# 乱数

import random

# カテゴリー変数のラベルエンコーディング

from sklearn.preprocessing import LabelEncoder

# ファイル管理

import os

import zipfile

# 警告の非表示

import warnings

warnings.filterwarnings('ignore')

# 可視化表示

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# 乱数の設定

SEED = 42



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



seed_everything(SEED)
PATH = '../input/competitive-data-science-predict-future-sales'

# 利用可能なファイルリスト

print(os.listdir(PATH))
# 訓練データの読み込み

sales_train = pd.read_csv(PATH+'/sales_train.csv')
def reduce_mem_usage(df, verbose=True):

    """

    データのメモリを減らすためにデータ型を変更する関数

    （引用元：https://www.kaggle.com/fabiendaniel/elo-world）

    （参考：https://qiita.com/hiroyuki_kageyama/items/02865616811022f79754）

    Param:

        df: DataFrame

        変換したいデータフレーム

        verbose: bool

        削減したメモリの表示

    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        # columns毎に処理

        col_type = df[col].dtypes

        if col_type in numerics:

            # numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
# データの読み込みとデータサイズの削減

sales_train = reduce_mem_usage(sales_train)

test = pd.read_csv(PATH+'/test.csv')

test = reduce_mem_usage(test)



sample_sub = pd.read_csv(PATH+'/sample_submission.csv')



items = pd.read_csv(PATH+'/items.csv')

items = reduce_mem_usage(items)

item_categories = pd.read_csv(PATH+'/item_categories.csv')

item_categories = reduce_mem_usage(item_categories)

shops = pd.read_csv(PATH+'/shops.csv')

shops = reduce_mem_usage(shops)
# 訓練データの表示確認

print(sales_train.shape)

sales_train.head()
# テストデータの表示確認

print(test.shape)

test.head()
# testのitem_idの種類の数とtrainのitem_idの種類の共通部分の要素数を取得 test['item_id']の要素数から引かれるのでsetとして作成

test_item_id_inter = len(set(test['item_id']).intersection(set(sales_train['item_id'])))

test_item_id_len = len(set(test['item_id']))



# testに存在してtrainにないitem_idを出力

print(test_item_id_len - test_item_id_inter)

# testの商品IDの数(重複は除く)

print(test_item_id_len)

# testの総数

print(len(test))
# submitfileの表示確認

print(sample_sub.shape)

sample_sub.head()
print(items.shape)

items.head()
items = items.drop(['item_name'], axis=1)
print(item_categories.shape)

item_categories.head()
item_categorie = item_categories['item_category_name'].unique()

item_categorie
# '-'でカテゴリ名を分割

item_categories['split'] = item_categories['item_category_name'].str.split('-')

# typeには-で分割した先頭の値を代入

item_categories['type'] = item_categories['split'].map(lambda x:x[0].strip())

# sub_typeには-で分割した2番目の値を代入、sub-typeには、typeのデータをsub_typeとして代入

item_categories['sub_type'] = item_categories['split'].map(lambda x:x[1].strip() if len(x) > 1 else x[0].strip())

item_categories.head()
# splitカラムの削除

item_categories.drop('split', axis=1, inplace=True)

# 'item_category_name'カラムの削除

item_categories.drop('item_category_name', axis=1, inplace=True)
item_categories['type'].value_counts()
item_categories['sub_type'].value_counts()
# # typeをone-hot encodingする

# types = pd.DataFrame(item_categories['type'])

# # one-hot encoding

# types = pd.get_dummies(types)



# # shops, city_onehotを横方向に連結

# item_categories = pd.concat([item_categories, types], axis=1)

# # shopsからcity_nameカラムを削除

# item_categories.drop('type', axis=1, inplace=True)

# item_categories.head()
# # sub_typeをone-hot encodingする

# sub_types = pd.DataFrame(item_categories['sub_type'])

# # one-hot encoding

# sub_types = pd.get_dummies(sub_types)



# # shops, city_onehotを横方向に連結

# item_categories = pd.concat([item_categories, sub_types], axis=1)

# # shopsからcity_nameカラムを削除

# item_categories.drop('sub_type', axis=1, inplace=True)

# item_categories.head()
# LabelEncoder

from sklearn.preprocessing import LabelEncoder



item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])

item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['sub_type'])

# item_categoriesからtypeとsub_typeカラムを削除

item_categories.drop('sub_type', axis=1, inplace=True)

item_categories.drop('type', axis=1, inplace=True)

item_categories.head()
print(shops.shape)

shops
# shop_idの統一

# マージ後を考え、`sales_train`と`test`に対してshop_id = 0 を shop_id = 57に shop_id = 1 を shop_id = 58 に shop_id = 10 を shop_id = 11に変換する

sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57

test.loc[test['shop_id'] == 0, 'shop_id'] = 57



sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58

test.loc[test['shop_id'] == 1, 'shop_id'] = 58



sales_train.loc[sales_train['shop_id'] == 10, 'shop_id'] = 11

test.loc[test['shop_id'] == 10, 'shop_id'] = 11
# shop_name先頭の!を削除

shops.loc[shops['shop_name'] == '!Якутск Орджоникидзе, 56 фран', 'shop_name'] = 'Якутск Орджоникидзе, 56 фран'

shops.loc[shops['shop_name'] == '!Якутск ТЦ "Центральный" фран', 'shop_name'] = 'кутск ТЦ "Центральный" фран'



# shop_nameの先頭を抽出してcity_nameを追加

shops['city_name'] = shops['shop_name'].str.split(' ').map(lambda x : x[0])    # 先頭から一番初めの半角スペースまでの文字列を抽出

shops.head()
# city_name = pd.DataFrame(shops['city_name'])

# # one-hot encoding

# city_onehot = pd.get_dummies(city_name)



# # shops, city_onehotを横方向に連結

# shops = pd.concat([shops, city_onehot], axis=1)

# # shopsからcity_nameカラムを削除

# shops.drop('city_name', axis=1, inplace=True)

# shops.head()
from sklearn.preprocessing import LabelEncoder

# LabelEncoder

shops['city_code'] = LabelEncoder().fit_transform(shops['city_name'])

# shopsからcity_nameカラムを削除

shops.drop('city_name', axis=1, inplace=True)

shops.head()
shops.drop('shop_name', axis=1, inplace=True)
# trainデータにて、'date_block_num','shop_id','item_id'でGROUP化したDataFrameGroupByオブジェクトに対して、'item_cnt_day'を集計

group = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

# 列名の更新

group.columns = ['item_cnt_month']

# DataFrameGroupBy -> DataFrame に変換

group.reset_index(inplace=True)

group.head()
# groups

groups = pd.DataFrame(group['item_cnt_month'])

# shops, city_onehotを横方向に連結

sales_train = pd.concat([sales_train, groups], axis=1)

sales_train.head()
# 欠損値計算関数

def missing_value_table(df):

    """欠損値の数とカラムごとの割合の取得

    Param : DataFrame

    確認を行うデータフレーム

    """

    # 欠損値の合計

    mis_val = df.isnull().sum()

    # カラムごとの欠損値の割合

    mis_val_percent = 100 * mis_val / len(df)

    # 欠損値の合計と割合をテーブルに結合

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # カラム名の編集

    mis_val_table = mis_val_table.rename(

        columns={0:'Missing Values', 1:'% of Total Values'}

    )

    # データを欠損値のあるものだけにし。小数点以下1桁表示で降順ソートする

    mis_val_table = mis_val_table[mis_val_table.iloc[:, 1] != 0].sort_values(

        '% of Total Values', ascending=False

    ).round(1)

    

    # 欠損値をもつカラム数の表示

    print('このデータフレームのカラム数は、', df.shape[1])

    print('このデータフレームの欠損値列数は、', mis_val_table.shape[0])

    

    # 欠損値データフレームを返す

    return mis_val_table
# 欠損値情報の表示

Missing_value = missing_value_table(sales_train)

Missing_value.head()
sales_train.isnull().sum()
# 欠損値情報の表示

Missing_value = missing_value_table(test)

Missing_value.head()
test.isnull().sum()
# 欠損値情報の表示

Missing_value = missing_value_table(items)

Missing_value.head()
# 欠損値情報の表示

Missing_value = missing_value_table(item_categories)

Missing_value.head()
# 欠損値情報の表示

Missing_value = missing_value_table(shops)

Missing_value.head()
# sales_train列ごとの型数と出現個数の確認

print('type:\n{}\n\nvalue counts:\n{}\n'.format(sales_train.dtypes, sales_train.dtypes.value_counts()))
# test列ごとの型数と出現個数の確認

print('type:\n{}\n\nvalue counts:\n{}\n'.format(test.dtypes, test.dtypes.value_counts()))
# items列ごとの型数と出現個数の確認

print('type:\n{}\n\nvalue counts:\n{}\n'.format(items.dtypes, items.dtypes.value_counts()))
# item_categories列ごとの型数と出現個数の確認

print('type:\n{}\n\nvalue counts:\n{}\n'.format(item_categories.dtypes, item_categories.dtypes.value_counts()))
# shops列ごとの型数と出現個数の確認

print('type:\n{}\n\nvalue counts:\n{}\n'.format(shops.dtypes, shops.dtypes.value_counts()))
print('before ', sales_train.shape)

# データの結合

train = pd.merge(sales_train, items, on='item_id', how='left')

print('after ', train.shape)

train.head()
print('before ', test.shape)

# データの結合

test = pd.merge(test, items, on='item_id', how='left')

print('after ', test.shape)

test.head()
print('before ', train.shape)

# データの結合

train = pd.merge(train, item_categories, on='item_category_id', how='left')

print('after ', train.shape)

train.head()
print('before ', test.shape)

# データの結合

test = pd.merge(test, item_categories, on='item_category_id', how='left')

print('after ', test.shape)

test.head()
print('before ', train.shape)

# データの結合

train = pd.merge(train, shops, on='shop_id', how='left')

print('after ', train.shape)

train.head()
print('before ', test.shape)

# データの結合

test = pd.merge(test, shops, on='shop_id', how='left')

print('after ', test.shape)

test.head()
# 欠損値情報の表示

Missing_value_train = missing_value_table(train)

Missing_value_train.head()
# 欠損値情報の表示

Missing_value_test = missing_value_table(test)

Missing_value_test.head()
# 型の確認

train.dtypes
# 型の確認

test.dtypes
# # 処理が重いので基本的にコメントアウト

# import pandas_profiling as pdp  # pandas_profilingのインポート

# pdp.ProfileReport(train)  # レポートの作成
# date_block_numは全体を+1

train['date_block_num'] += 1

train['date_block_num']



# 2015年11月のデータのためdate_block_num = 34として列を追加

test['date_block_num'] = 34

# 型のキャスト

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)

test.head()
# item_priceとitem_cnt_dayの箱ひげ図表示

plt.ylim(-100, 310000)

sns.boxplot(y='item_price', data=train)

plt.grid()

plt.tight_layout()

plt.show()

plt.ylim(-100, 2500)

plt.grid()

plt.tight_layout()

sns.boxplot(y='item_cnt_day', data=train)
# 外れ値の除外

print('before ', train.shape)

train = train[train['item_price'] < 100000]

train = train[train['item_cnt_day'] < 1200]

print('after ', train.shape)
# item_priceが-1のデータ数

print(train[train['item_price'] == -1]['item_price'].value_counts())

# item_priceの-1の値を持つデータの確認

train[train['item_price'] == -1]
# item_id 2973のときのitem_price出現個数の確認

train.loc[train['item_id'] == 2973, 'item_price'].value_counts()
# 円グラフで可視化

plt.figure(figsize=(8, 8))

plt.pie(

    train.loc[train['item_id'] == 2973, 'item_price'].value_counts(),    # データの出現頻度

    labels=train.loc[train['item_id'] == 2973, 'item_price'].value_counts().index,    # ラベル名の指定

    counterclock=False,    # データを時計回りに入れる

    startangle=90,          # データの開始位置 90の場合は円の上から開始

    autopct='%1.1f%%',      # グラフ内に構成割合のラベルを小数点1桁まで表示

    pctdistance=0.8         # ラベルの表示位置

)

plt.tight_layout()

plt.show()
# item_id 2973のときの訓練データを抽出

item_id_2973 = train[train['item_id'] == 2973]

# item_id_2973の中で、shop_id 32のデータを抽出し、出現個数を確認

item_id_2973.loc[item_id_2973['shop_id']==32, 'item_price'].value_counts()
# 円グラフで可視化

plt.figure(figsize=(8, 8))

plt.pie(

    item_id_2973.loc[item_id_2973['shop_id']==32, 'item_price'].value_counts(),    # データの出現頻度

    labels=item_id_2973.loc[item_id_2973['shop_id']==32, 'item_price'].value_counts().index,    # ラベル名の指定

    counterclock=False,    # データを時計回りに入れる

    startangle=90,          # データの開始位置 90の場合は円の上から開始

    autopct='%1.1f%%',      # グラフ内に構成割合のラベルを小数点1桁まで表示

    pctdistance=0.8         # ラベルの表示位置

)

plt.tight_layout()

plt.show()
# item_priceの-1の値を2499.0に置き換える

train[train['item_price'] == -1] = 2499
# 置き換えが完了したかの確認

train[train['item_price'] == -1]
train.dtypes
# object型の確認

train.select_dtypes(include=object).head()
# object型の確認

test.select_dtypes(include=object).head()
# object型の`date`を削除

train.drop('date', axis=1 ,inplace=True)
# 説明変数item_priceがtestより多い

train.shape
test.shape
train['item_price']
# trainデータにて、'shop_id','item_id'でGROUP化したDataFrameGroupByオブジェクトに対して、'item_price'の平均

group_item_price = train.groupby(['shop_id','item_id']).agg({'item_price': ['mean']})

# 列名の更新

group_item_price.columns = ['item_price']

# DataFrameGroupBy -> DataFrame に変換

group_item_price.reset_index(inplace=True)

group_item_price.head()
print('before ', test.shape)

# 'shop_id', 'item_id'のデータを結合

test = pd.merge(test, group_item_price, on=['shop_id', 'item_id'], how='left')

print('after ', test.shape)

test.head()
# item_priceの欠損値を中央値で補完

test['item_price'] = test['item_price'].fillna(test['item_price'].median())
# 説明変数item_priceがtestより多い

train.shape
test.shape
# 目的変数と説明変数に分割

y_train = train['item_cnt_month']    # 目的変数

X_train = train.drop('item_cnt_month', axis=1)    # 訓練データの説明変数

X_test = test
from sklearn.model_selection import train_test_split



# train:valid = 7:3

X_train, X_valid, y_train, y_valid = train_test_split(

    X_train,             # 対象データ1

    y_train,             # 対象データ2

    test_size=0.3,       # 検証用データを3に指定

#     stratify=y_train,    # 訓練データで層化抽出

    random_state=42

)
# LightGBMで学習の実施

import lightgbm as lgb



# カテゴリー変数をリスト形式で宣言(A-Z順で宣言する)

categorical_features = ['city_code', 'type_code', 'subtype_code']



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

        'objective': 'regression',    # 回帰問題

        'metric': 'rmse',      # RMSE (平均二乗誤差平方根) の最小化を目指す

        'learning_rate': 0.1, # 学習率

        'max_depth': -1, # 木の数 (負の値で無制限)

        'num_leaves': 9, # 枝葉の数

        'drop_rate': 0.15,

        'verbose': 0

    }



lgb_model = lgb.train(

    params,    # パラメータ

    lgb_train,    # 学習用データ

    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ

    verbose_eval=10,    # 検証データは10個

    num_boost_round=1000,    # 学習の実行回数の最大値

    early_stopping_rounds=100    # 連続25回学習で検証データの性能が改善しない場合学習を打ち切る

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
# 予測データをcsvに変換

sub = sample_sub = pd.read_csv(PATH+'/sample_submission.csv')    # サンプルの予測データ

sub['item_cnt_month'] = lgb_y_pred



sub.to_csv('./submit_lightgbm.csv', index=False)

sub.head()
# object型の`date`を削除

train.drop(['item_id', 'type_code'], axis=1 ,inplace=True)

# object型の`date`を削除

test.drop(['item_id', 'type_code'], axis=1 ,inplace=True)
# 目的変数と説明変数に分割

y_train = train['item_cnt_month']    # 目的変数

X_train = train.drop('item_cnt_month', axis=1)    # 訓練データの説明変数

X_test = test
# train:valid = 7:3

X_train, X_valid, y_train, y_valid = train_test_split(

    X_train,             # 対象データ1

    y_train,             # 対象データ2

    test_size=0.3,       # 検証用データを3に指定

#     stratify=y_train,    # 訓練データで層化抽出

    random_state=42

)
# カテゴリー変数をリスト形式で宣言

categorical_features = ['city_code', 'subtype_code']



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

        'objective': 'regression',    # 回帰問題

        'metric': 'rmse',      # RMSE (平均二乗誤差平方根) の最小化を目指す

    }



lgb_model = lgb.train(

    params,    # パラメータ

    lgb_train,    # 学習用データ

    valid_sets=[lgb_train, lgb_valid],    # 訓練中に評価されるデータ

    verbose_eval=10,    # 検証データは10個

    num_boost_round=1000,    # 学習の実行回数の最大値

    early_stopping_rounds=25    # 連続25回学習で検証データの性能が改善しない場合学習を打ち切る

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
## 推論                 

lgb_y_pred2 = lgb_model.predict(

    X_test,    # 予測を行うデータ

    num_iteration=lgb_model.best_iteration, # 繰り返しのインデックス Noneの場合、best_iterationが存在するとダンプされます。それ以外の場合、すべての繰り返しがダンプされます。 <= 0の場合、すべての繰り返しがダンプされます。

)

# 結果の表示

lgb_y_pred2[:10]
# 予測データをcsvに変換

sub2 = sample_sub = pd.read_csv(PATH+'/sample_submission.csv')    # サンプルの予測データ

sub2['item_cnt_month'] = lgb_y_pred



sub2.to_csv('./submit_lightgbm_drop_futer.csv', index=False)

sub2.head()