# General imports

import numpy as np

import pandas as pd

import os, sys, gc, time, warnings, pickle, psutil, random



from math import ceil



from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')
## メモリ使用量を確認するためのシンプルな「メモリプロファイラ」

def get_memory_usage():

    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 

        

def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)
## メモリ削減

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
## dtypesを失わないための連結による結合

def merge_by_concat(df1, df2, merge_on):

    merged_gf = df1[merge_on]

    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')

    new_columns = [col for col in list(merged_gf) if col not in merge_on]

    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)

    return df1
########################### Vars

#################################################################################

TARGET = 'sales'         # Our main target

END_TRAIN = 1913         # Last day in train set

MAIN_INDEX = ['id','d']  # We can identify item by these columns
########################### Load Data

#################################################################################

print('Load Main Data')



# Here are reafing all our data 

# without any limitations and dtype modification

train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

prices_df = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
########################### グリッド作成

#################################################################################

print('Create Grid')



# 水平表現を変形できます 

# 縦方向の「ビュー」 

# index は「id」、「item_id」、「dept_id」、「cat_id」、「store_id」、「state_id」 

# ラベルは「d_」列です



index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

grid_df = pd.melt(train_df, 

                  id_vars = index_columns, 

                  var_name = 'd', 

                  value_name = TARGET)



# train_dfを見ると、行は多くありません 

# しかし、それぞれの日により多くのtrain dataを提供できます

print('Train rows:', len(train_df), len(grid_df))



# 予測できるように

# "test set"をグリッドに追加する必要があります

add_grid = pd.DataFrame()

for i in range(1,29):

    temp_df = train_df[index_columns]

    temp_df = temp_df.drop_duplicates()

    temp_df['d'] = 'd_'+ str(END_TRAIN+i)

    temp_df[TARGET] = np.nan

    add_grid = pd.concat([add_grid,temp_df])



grid_df = pd.concat([grid_df,add_grid])

grid_df = grid_df.reset_index(drop=True)



# 一時的なDFを削除する

del temp_df, add_grid



# オリジナルのtrain_dfは必要ありません もう削除できます

del train_df



# df = df構築を使用する必要はありません 

# 代わりにinplace = Trueを使用できます。 

# 次のようにして

# grid_df.reset_index（drop = True、inplace = True） 



# メモリ使用量を確認しましょう

print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))



# 一部のメモリを解放できます 

# 文字列をカテゴリに変換する 

# 結合には影響せず、貴重なデータを失うことはありません

for col in index_columns:

    grid_df[col] = grid_df[col].astype('category')



# メモリ使用量をもう一度確認してみましょう

print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
########################### 製品発売日

#################################################################################

print('Release week')



# 各train_dfアイテム行の先行ゼロ値は実際の0売上ではなく、

# 店にアイテムがないことを意味します

# このようなゼロを削除することで、一部のメモリを安全にすることができます



# 価格は週ごとに設定されるので、 リリース週があまり正確ではありません

release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()

release_df.columns = ['store_id','item_id','release']



# これでrelease_dfを結合できます

grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])

del release_df



# ＃grid_dfから「ゼロ」行をいくつか削除したい  

# それを行うには、wm_yr_wk列が必要です 

# 部分的にcalendar_dfを結合してみましょう

grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])

                      

# これで、いくつかの行をカットして安全なメモリにできます

grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]

grid_df = grid_df.reset_index(drop=True)



# メモリ使用量を確認しましょう

print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))



# 特徴量の1つとしてリリース週を維持する必要がありますか？ 

# 良いCVだけが答えを出すことができます。 

# リリース値を縮小してみましょう。 

# 最小変換はここでは役に立たない

# int16→integer（-32768から32767）

# やgrid_df ['release'].max（）→int16のような変換は。

# しかし、必要な場合に備えて、変換するある方法があります。

grid_df['release'] = grid_df['release'] - grid_df['release'].min()

grid_df['release'] = grid_df['release'].astype(np.int16)



# メモリ使用量をもう一度確認してみましょう

print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
########################### 保存 part 1

#################################################################################

print('Save Part 1')



# BASEグリッドの準備ができました 

# 今後の使用のため（モデルトレーニング）pickleファイルとして保存できます

grid_df.to_pickle('grid_part_1.pkl')



print('Size:', grid_df.shape)
########################### 価格

#################################################################################

print('Prices')



# 基本的な集計を行うことができます

prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')

prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')

prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')

prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')



# そして価格正規化を行います（最小/最大スケーリング）

prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']



# 一部のアイテムはインフレに依存する可能性があります 

# いくつかのアイテムは非常に「安定」しています

prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')

prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')



# 移動集計をしたい 

# 「枠」として月と年が欲しい

calendar_prices = calendar_df[['wm_yr_wk','month','year']]

calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])

prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')

del calendar_prices



# これで、価格に（ある種の）「勢い」を加えることができます 

# 週ごとにシフト 

# 月平均 年平均

prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))

prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')

prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')



del prices_df['month'], prices_df['year']
########################### 価格の結合と保存 part 2

#################################################################################

print('Merge prices and save part 2')



# 価格の結合

original_columns = list(grid_df)

grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')

keep_columns = [col for col in list(grid_df) if col not in original_columns]

grid_df = grid_df[MAIN_INDEX+keep_columns]

grid_df = reduce_mem_usage(grid_df)



# 保存 part 2

grid_df.to_pickle('grid_part_2.pkl')

print('Size:', grid_df.shape)



# prices_dfはもういらない

del prices_df



# 新しい列を削除できます 

# または単にpart_1をロードする

grid_df = pd.read_pickle('grid_part_1.pkl')
########################### カレンダーの結合

#################################################################################

grid_df = grid_df[MAIN_INDEX]



# カレンダーを部分的に結合

icols = ['date',

         'd',

         'event_name_1',

         'event_type_1',

         'event_name_2',

         'event_type_2',

         'snap_CA',

         'snap_TX',

         'snap_WI']



grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')



# データを縮小する 

# 'snap_'列はboolまたはint8に変換できる 

icols = ['event_name_1',

         'event_type_1',

         'event_name_2',

         'event_type_2',

         'snap_CA',

         'snap_TX',

         'snap_WI']

for col in icols:

    grid_df[col] = grid_df[col].astype('category')



# 日時に変換

grid_df['date'] = pd.to_datetime(grid_df['date'])



# 日付からいくつかの特徴量を作る

grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)

grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)

grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)

grid_df['tm_y'] = grid_df['date'].dt.year

grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)

grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)



grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)

grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)



# 日付の削除

del grid_df['date']
########################### 保存 part 3 (日付)

#################################################################################

print('Save part 3')



# 保存 part 3

grid_df.to_pickle('grid_part_3.pkl')

print('Size:', grid_df.shape)



# calendar_dfはもういらない

del calendar_df

del grid_df
########################### 追加のクリーニング

#################################################################################



## Part 1

# 'd' → int

grid_df = pd.read_pickle('grid_part_1.pkl')

grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)



# 'wm_yr_wk'の削除

# テスト値がtrainにないため

del grid_df['wm_yr_wk']

grid_df.to_pickle('grid_part_1.pkl')



del grid_df
########################### 概要

#################################################################################



# 今、私たちは3つの特徴量セットを持っています

grid_df = pd.concat([pd.read_pickle('grid_part_1.pkl'),

                     pd.read_pickle('grid_part_2.pkl').iloc[:,2:],

                     pd.read_pickle('grid_part_3.pkl').iloc[:,2:]],

                     axis=1)

                     

# メモリ使用量をもう一度

print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

print('Size:', grid_df.shape)



# 2.5GiB + はまだ大きすぎて、（Kaggleでは）モデルをトレーニングできません 

# lag特徴量はまだありません 

# しかし、state_idまたはshop_idでトレーニングできるとしたらどうでしょう？

state_id = 'CA'

grid_df = grid_df[grid_df['state_id']==state_id]

print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

#           Full Grid:   1.2GiB



store_id = 'CA_1'

grid_df = grid_df[grid_df['store_id']==store_id]

print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

#           Full Grid: 321.2MiB



# もう十分だと思います 

# 他のNotebookではlag特徴量について話します

# Thank you.
########################### 特徴量の最終リスト

#################################################################################

grid_df.info()