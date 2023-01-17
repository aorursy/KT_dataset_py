import numpy as np 
import pandas as pd 
import os

#可視化系
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
#matplotlibのグラフのデザインかえれるんすねーはー知らんかった
#https://qiita.com/eriksoon/items/b93030ba4dc686ecfbba　ここ参照
plt.style.use('ggplot')

#定番の機械学習系
import lightgbm as lgb
import xgboost as xgb

#定番の時間
import time
import datetime

#これもよく見るけどダミー変数作成の時に使いますね
from sklearn.preprocessing import LabelEncoder
#簡単にトレーニングデータとテストデータを分けれるやつ
from sklearn.model_selection import StratifiedKFold, KFold
#これもよく見るけど平均二乗誤差 (MSE)
#確かこのコンペのスコア算出法は二乗平均平方根誤差 (RMSE)
#https://pythondatascience.plavox.info/scikit-learn/%E5%9B%9E%E5%B8%B0%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E8%A9%95%E4%BE%A1
#ここ分かり易い
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
#カーネル必須ガベージコレクション
import gc
#catboostは珍しい様な？
from catboost import CatBoostRegressor

#よく見るけどmatplotlibより綺麗で3Dモデルも書けるから使ってる？？
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#何回かみたけどアラートっぽい　
import warnings
warnings.filterwarnings("ignore")

pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)

# import workalendar
# from workalendar.america import Brazil
%%time
#セルの実行時間測れます　一番上に記述しないとエラー吐くよ
#訓練データを読み込ませる
train = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])
#テストデータを読み込ませる
test = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])
#サンプル提出データを読み込ませる
submission = pd.read_csv('../input/sample_submission.csv')
#メモリ使用量を減らす関数を定義
def reduce_mem_usage(df, verbose=True):
    #型宣言
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    #メモリの消費割合を格納
    start_mem = df.memory_usage().sum() / 1024**2
    #カラム名を一つずつ引っ張ってくる
    for col in df.columns:
        #カラムの型をcol_typeに代入
        col_type = df[col].dtypes
        #型が数字だったら
        if col_type in numerics:
            #c_minの最低値をc_maxに最大値を入れる
            c_min = df[col].min()
            c_max = df[col].max()
            #型名の最初3文字がint~型だったら
            if str(col_type)[:3] == 'int':
                #c_minがint8の数値範囲にあればint8型で格納する
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                #同様にint16で
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                #同様にint32
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                #同様にint64
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                 #同様にfloat16
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                #同様にfloat32
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    #同様にfloat64
                    df[col] = df[col].astype(np.float64)    
    #メモリの消費割合の算出
    end_mem = df.memory_usage().sum() / 1024**2
    #算出？？？？わからん
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
#データディクショナリのエクセル（カラム情報かいてるとこ）
#の訓練データ情報を開いて見てる
e = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='train')
e

#カラム名
#英語名　　　　　　　　　日本語名
#card_id　　　　　　　　カードID　　　　
#first_active_month　　初購入月　　　　
#feature_1            #データディクショナリのエクセル（カラム情報かいてるとこ）
#の訓練データ情報を開いて見てる
e = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='train')
e

#カラム名
#英語名　　　　　　　　　日本語名
#card_id　　　　　　　　カードID　　　　
#first_active_month　　初購入月　　　　
#feature_1#データディクショナリのエクセル（カラム情報かいてるとこ）
#の訓練データ情報を開いて見てる
e = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='train')
e

#カラム名
#英語名　　　　　　　　　日本語名
#card_id　　　　　　　　カードID　　　　
#first_active_month　　初購入月　　　　
#feature_1            匿名カードの分類機能(ゴールド会員？)
#feature_2　　　　　　　３段階に分かれている
#feature_3
#target　　　　　　　　　履歴および評価期間の2ヶ月後に算出された
#　　　　　　　　　　　　　ロイヤリティ数値スコア？？
#クレカ顧客のカテゴリ分けの型を'category'型に変換(文字列)
train['feature_1'] = train['feature_1'].astype('category')
train['feature_2'] = train['feature_2'].astype('category')
train['feature_3'] = train['feature_3'].astype('category')
train.head()
train.info()
#クレカ顧客のカテゴリ分け毎に
#target　　　　　　　　　履歴および評価期間の2ヶ月後に算出された
#　　　　　　　　　　　　　ロイヤリティ数値スコア？？
#ターゲット(上参照)の分布度合いをヴァイオリンプロットで可視化
#一行目は図のレイアウト設定の宣言　多分３つ横並び図を作るのにこの書き方じゃないとダメ
fig, ax = plt.subplots(1, 3, figsize = (16, 6))
plt.suptitle('Violineplots for features and target');
sns.violinplot(x="feature_1", y="target", data=train, ax=ax[0], title='feature_1');
sns.violinplot(x="feature_2", y="target", data=train, ax=ax[1], title='feature_2');
sns.violinplot(x="feature_3", y="target", data=train, ax=ax[2], title='feature_3');
#それぞれのカテゴリ別の顧客数
fig, ax = plt.subplots(1, 3, figsize = (16, 6));
train['feature_1'].value_counts().sort_index().plot(kind='bar', ax=ax[0], color='teal', title='feature_1');
train['feature_2'].value_counts().sort_index().plot(kind='bar', ax=ax[1], color='brown', title='feature_2');
train['feature_3'].value_counts().sort_index().plot(kind='bar', ax=ax[2], color='gold', title='feature_3');
plt.suptitle('Counts of categiories for features');
#テストデータの型変換も行っている
test['feature_1'] = test['feature_1'].astype('category')
test['feature_2'] = test['feature_2'].astype('category')
test['feature_3'] = test['feature_3'].astype('category')
train.head()
#first_active_month　　初購入月
#このカラムをsort_indexで日付（行を）昇順にソートして各々の日付の数を算出
d1 = train['first_active_month'].value_counts().sort_index()
d1.head()
#上の処理をテストデータも同様に行っている
d2 = test['first_active_month'].value_counts().sort_index()
#折れ線グラフの表記
#x軸とy軸と折れ線の指定
data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]
#タイトル
layout = go.Layout(dict(title = "Counts of first active",
                  #x軸の項目名
                  xaxis = dict(title = 'Month'),
                  #ｙ軸の項目名
                   yaxis = dict(title = 'Count'),
                 #凡例orientationは配置方法だったような
                  ),legend=dict(
                orientation="v"))
#上の処理でデータ入れて　このコードでiplot(多分plotlyがこれ)の可視化実行
py.iplot(dict(data=data, layout=layout))
#first_active_month　　初購入月の値が空値でfeature_1分類が5でfeature_2が2でfeature_3が１だったら
#初購入月に最小値を入れる
test.loc[test['first_active_month'].isna(),'first_active_month'] = test.loc[(test['feature_1'] == 5) & (test['feature_2'] == 2) & (test['feature_3'] == 1), 'first_active_month'].min()
#target　　　　　　　　　履歴および評価期間の2ヶ月後に算出された
#　　　　　　　　　　　　　ロイヤリティ数値スコア？？
#上記の分布をヒストグラム可視化
#謎の-30
plt.hist(train['target']);
plt.title('Target distribution');
#-20以下のターゲットの数を数えて表示
print('There are {0} samples with target lower than -20.'.format(train.loc[train.target < -20].shape[0]))
#max_date変数に　#first_active_month　　初購入月　の最終日だけを抜き出して放り込む
max_date = train['first_active_month'].dt.date.max()
def process_main(df):
    #年、月、日に分ける？
    date_parts = ["year", "weekday", "month"]
    #１個ずつfor文回す感じ？
    for part in date_parts:
        #_でデータ分割
        part_col = 'first_active_month' + "_" + part
        #datetime型からint型に変換
        df[part_col] = getattr(df['first_active_month'].dt, part).astype(int)
    #月の最終日からどれ位前かを経過時間カラム名として追加        
    df['elapsed_time'] = (max_date - df['first_active_month'].dt.date).dt.days
    
    return df
#わかりません
train = process_main(train)
test = process_main(test)
#今度はData_Dictionaryのシート名がhistoryのデータを読み込む
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
e = pd.read_excel('../input/Data_Dictionary.xlsx', sheet_name='history')
e

#2カードID
#3現在から何か月前
#4購入日
#5決済まで終わったか　YかN
#6カテゴリ３　A~C
#7分割払い数
#8カテゴリ１ YかN
#9販売側カテゴリID
#10 サブセクターID わかりません
#11 販売側ID
#12 購入数 これ意味不明
#13 市街ID
#14　国ID
#15 カテゴリ２　1~4
#レコード件数表示
print(f'{historical_transactions.shape[0]} samples in data')
historical_transactions.head()
#5決済まで終わったか　YかNのバイナリ値に変換しましょう
#Yだったら1いれてNだったら0にするラムダ式いれてます。
historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].apply(lambda x: 1 if x == 'Y' else 0)
#決済完了率の表示　8%位は決済失敗してる
print(f"At average {historical_transactions['authorized_flag'].mean() * 100:.4f}% transactions are authorized")
historical_transactions['authorized_flag'].value_counts().plot(kind='barh', title='authorized_flag value counts');
#最低ランクの方が重要で30回に1回しか成功してない　なんで？　不正取引？
autorized_card_rate = historical_transactions.groupby(['card_id'])['authorized_flag'].mean().sort_values()
autorized_card_rate.head()
#後ろは当然100%
autorized_card_rate.tail()
historical_transactions['installments'].value_counts()
#分割払いと決済完了率で集計して平均
historical_transactions.groupby(['installments'])['authorized_flag'].mean()
#installmentsをcategory型に型変換
historical_transactions['installments'] = historical_transactions['installments'].astype('category')
#purchase_date 購入日に時間も書いてあるので　日付だけに変換
historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
plt.title('Purchase amount distribution.');
historical_transactions['purchase_amount'].plot(kind='hist');
#－１と０でループ
for i in [-1, 0]:
    #purchase_amount購入量が-1と0より下の値の数を抜き出す
    n = historical_transactions.loc[historical_transactions['purchase_amount'] < i].shape[0]
    #プリント
    print(f"There are {n} transactions with purchase_amount less than {i}.")
#次は0,10,100
for i in [0, 10, 100]:
    #今度は逆に上
    n = historical_transactions.loc[historical_transactions['purchase_amount'] > i].shape[0]
    print(f"There are {n} transactions with purchase_amount more than {i}.")
#０以下の分布具合を可視化
plt.title('Purchase amount distribution for negative values.');
historical_transactions.loc[historical_transactions['purchase_amount'] < 0, 'purchase_amount'].plot(kind='hist');
#置換する値を定義
map_dict = {'Y': 0, 'N': 1}
#ラムダ文使ってカテゴリ１の置換実行
historical_transactions['category_1'] = historical_transactions['category_1'].apply(lambda x: map_dict[x])
#カテゴリ１を集計して　purchase_amount 購入量の平均、標準偏差、カウント数と　authorized_flag 決済完了率の平均、標準偏差を表示　これ便利だねぇ
historical_transactions.groupby(['category_1']).agg({'purchase_amount': ['mean', 'std', 'count'], 'authorized_flag': ['mean', 'std']})
#ループ文でカラムまわし
for col in ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']:
    #各々のカラムの一意の値をカウントしてプリント
    print(f"There are {historical_transactions[col].nunique()} unique values in {col}.")
#取引履歴集計
def aggregate_historical_transactions(trans, prefix):
    #purchase_month　購入日付の月だけ抜き出してる
    trans['purchase_month'] = trans['purchase_date'].dt.month
#     trans['year'] = trans['purchase_date'].dt.year
#     trans['weekofyear'] = trans['purchase_date'].dt.weekofyear
#     trans['month'] = trans['purchase_date'].dt.month
#     trans['dayofweek'] = trans['purchase_date'].dt.dayofweek
#     trans['weekend'] = (trans.purchase_date.dt.weekday >=5).astype(int)
#     trans['hour'] = trans['purchase_date'].dt.hour
    #何か月違うかを表記(今日の日付ー購入日付)÷30の切り捨てで計算
    trans['month_diff'] = ((datetime.datetime.today() - trans['purchase_date']).dt.days)//30
    #ここは意味不明だが
    #month_lagは2018年の2月からどれだけ前の購入月か記述してるが
    #下記のコードを実行するとおかしくならない？と思う
    trans['month_diff'] += trans['month_lag']
    #分割払いのカラムをint型変換
    trans['installments'] = trans['installments'].astype(int)
    #purchase_date 購入日付を時系列インデックスに変換するみたいだけど
    trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(trans['purchase_date']). \
                                        astype(np.int64) * 1e-9
    #左の10の-9乗してる意味がさｐｐｐっぱり
    #ダミー変数作成メソッド(男→１　女→０みたいな置換)
    trans = pd.get_dummies(trans, columns=['category_2', 'category_3'])
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean', 'sum'],
        'category_2_2.0': ['mean', 'sum'],
        'category_2_3.0': ['mean', 'sum'],
        'category_2_4.0': ['mean', 'sum'],
        'category_2_5.0': ['mean', 'sum'],
        'category_3_1': ['sum', 'mean'],
        'category_3_2': ['sum', 'mean'],
        'category_3_3': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        #ptpは値の範囲を算出してくれる
        'purchase_date': [np.ptp, 'max', 'min'],
        'month_lag': ['min', 'max'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],
        'city_id': ['nunique'],
        'month_diff': ['min', 'max', 'mean']
    }
    #aggは指定したカラム毎に集計をとってくれる
    agg_trans = trans.groupby(['card_id']).agg(agg_func)
    #
    agg_trans.columns = [prefix + '_'.join(col).strip() for col in agg_trans.columns.values]
    #インデックスの番号降りなおし
    agg_trans.reset_index(inplace=True)

    df = (trans.groupby('card_id')
          .size()
          .reset_index(name='{}transactions_count'.format(prefix)))

    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')

    return agg_trans
#月間集計　引数history
def aggregate_per_month(history):
    #カードIDと何か月前に購入したかで集計
    grouped = history.groupby(['card_id', 'month_lag'])
    #installments 分割払いをint型に型変換
    history['installments'] = history['installments'].astype(int)
    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    
    return final_group

final_group = aggregate_per_month(historical_transactions) 
%%time
#いらんデータ削除
del d1, d2, autorized_card_rate
#軽くするためにガベージコレクション
gc.collect()
historical_transactions.head()
#historical_transactions 取引履歴シートに
#メモリ使用量を減らす関数を定義
#def reduce_mem_usage(df, verbose=True):
#上記のメソッド放り込み
historical_transactions = reduce_mem_usage(historical_transactions)
historical_transactions.head()
history = aggregate_historical_transactions(historical_transactions, prefix='hist_')
history = reduce_mem_usage(history)
gc.collect()


