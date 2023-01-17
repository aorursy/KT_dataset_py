import numpy as np 
import pandas as pd 
#可視化系
import matplotlib.pyplot as plt
import seaborn as sns
#機械学習
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
#よく見るけど未だわからん
import warnings
import time
#osと同じ　メモリ使用量とかも見れる
import sys
import datetime
#可視化　ダブってね？
import matplotlib.pyplot as plt
import seaborn as sns
#RMSE出すために使う
from sklearn.metrics import mean_squared_error
#サイキットラーンで使う（回帰？ナイーフベイズ？）
from sklearn.linear_model import BayesianRidge
#warningとセットで見る
warnings.simplefilter(action='ignore', category=FutureWarning)
#ガベージコレクション
import gc
#このカーネルでは使ってないけれど一応読み込み　解説
merchants = pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv')
merchants.head()
#データ読み込み parse_dateは指定したカラムをタイムスタンプ型として読み込む　今回はperchase_date=購入日を使用している
new_transactions = pd.read_csv('../input/elo-merchant-category-recommendation/new_merchant_transactions.csv', parse_dates=['purchase_date'])
historical_transactions = pd.read_csv('../input/elo-merchant-category-recommendation/historical_transactions.csv', parse_dates=['purchase_date'])
new_transactions.head()
historical_transactions.head()
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
#authorized_flag 決済完遂フラグとカテゴリ１　カラムをダミー変数化
#こっちはメソッド作り　Yを１　Nを０に置換
def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df
#new_merchant_transactions.csv(販売側の新規取引履歴)
#とhistorical_transactions.csv(カード会社側取引履歴)を上のメソッドでダミー変数化
historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)
%%time
#↑はメモリ使用量計測
#trainデータの整形メソッド
def read_data(input_file):
    #csvをdfに格納
    df = pd.read_csv(input_file)
    #trainデータの初購入月(first_active_month) をタイムスタンプ型に変換
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    # 2018年2月1日(基準日っぽい)ー初購入月(first_active_month)の日付　で日付だけをelapsed_timeに格納
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
#trainデータとtestデータの読み込み
train = read_data('../input/elo-merchant-category-recommendation/train.csv')
test = read_data('../input/elo-merchant-category-recommendation/test.csv')
#trainデータのtarget　　　　　　　　　履歴および評価期間の2ヶ月後に算出された
#　　　　　　　　　　　　　　　　　　　ロイヤリティ数値スコア？？をtarget変数に格納して

target = train['target']
#trainデータのtargetカラムは削除している
del train['target']
#ガベージコレクション
gc.collect()
train.head()
%%time
#historical_transactions.csv	クレカ取引履歴のカテゴリ２と３をダミー変数化する
historical_transactions = pd.get_dummies(historical_transactions, columns=['category_2', 'category_3'])
#category_2_1.0やcategory_3_Aができて０か１が入っている
historical_transactions.head()
#上記と同様にnew_merchant_transactions.csv 販売側の新規取引履歴のカテゴリ２と３
new_transactions = pd.get_dummies(new_transactions, columns=['category_2', 'category_3'])
new_transactions.head()
#historical_transactionsとnew_transactionsをメモリ使用量を減らすメソッドにかけてる
historical_transactions = reduce_mem_usage(historical_transactions)
new_transactions = reduce_mem_usage(new_transactions)
#メモリー使用量が1304Mb減少(54.8%)　newは84.24Mb
#クレカ取引履歴(historical_transactions.csv)の
#authorized_flag=カード決済完了フラグの合計と平均をagg_fun変数に格納
agg_fun = {'authorized_flag': ['sum', 'mean']}
#そのagg_funを使って　クレカ取引履歴(historical_transactions.csv)のカードID(card_id)を主キーとして
#グループバイ
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)

#ガベージコレクション
gc.collect()
auth_mean.head()
#上記のカラム名を引っ張ってきて　stripで空白削除して_でタイトルと結合
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.head()
#リセットインデックス（番号を↑から降りなおし）
auth_mean.reset_index(inplace=True)
auth_mean.head()
#authorized_flag=カード決済完了フラグが
#完了の１をauthorized_transaxtionsへ
#失敗の０をhistorical_transaxtionsに分けてデータ格納
authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
historical_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]
#ガベージコレクション
gc.collect()
authorized_transactions.head()
historical_transactions.head()
%%time
#上記の２つのデータの購入日(purchase_date)の月だけを抜き出して購入月(purchase_month)作成
historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
authorized_transactions['purchase_month'] = authorized_transactions['purchase_date'].dt.month
#new_merchant_transactions.csv(販売側の新規取引履歴)も購入日から購入月作成
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month
#ガベージコレクション
gc.collect()
%%time
def aggregate_transactions(history):
    #purchase_dateを時系列のインデックスとしてint64に型変換してるだろうというのはわかるが
    #1e-9=0.000000001がわからない（9桁で表示とか？）
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    #agg_func変数に作成したい項目を追加(np.ptpは値の範囲？)
    agg_func = {
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['min', 'max']
        }
    #引数の渡されたデータをagg_funncにカードID(card_id)を主キーにしてグループバイ
    agg_history = history.groupby(['card_id']).agg(agg_func)
    #カラム名をタイトルと結合
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    #リセットインデックス
    agg_history.reset_index(inplace=True)
    
    #カードID(card_id)グループバイしたのをsizeで要素数？？？を取得して
    #リセットインデックス(transactions_count)という項目名で
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    #agg_historyに上記のdfとその上のagg_historyをカードIDで外部結合する
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    #agg_historyを返す
    return agg_history
#ガベージコレクション
gc.collect()
%%time
#上記のメソッドをクレカ取引履歴(historical_transactions.csv)にかけてる
history = aggregate_transactions(historical_transactions)
history.head()
#正直transactions_countとpurchase_dateがなんでこうなってるのか
#未だ分かりません・・・
#分かり辛いが　カードID(card_id）以外のカラム名の先頭にhist_をつける
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
history[:5]
#ガベージコレクション
gc.collect()
%%time
#上記と同じ事をカード決済完了フラグが完了になっていたデータで行う
authorized = aggregate_transactions(authorized_transactions)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
authorized[:5]
gc.collect()
%%time
#上のaggregate_transactionsメソッドを#new_merchant_transactions.csv(販売側の新規取引履歴)にかけてる
new = aggregate_transactions(new_transactions)
#上記のauth_をnew_にしてるだけ
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
gc.collect()
new[:5]
%%time

def aggregate_per_month(history):
    #渡されたデータのカードID(card_id)とmonth_lag=基準日(2018年2月？）までの月差でグループバイ
    grouped = history.groupby(['card_id', 'month_lag'])

    #agg_funcに欲しい式情報を入れてる
    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }
    #上記のagg_funkをかけてる
    intermediate_group = grouped.agg(agg_func)
    #カラム名を空白削除と先頭にタイトル名_の形で結合
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    #リセットインデックス
    intermediate_group.reset_index(inplace=True)
    #上記のデータをカードID(card_id)の平均と標準偏差でグループバイ
    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    #カラム名を同様に変更
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)
    #この値を返す
    return final_group
#___________________________________________________________
#historical_transactions.csv	クレカ取引履歴を上記のメソッドにかける
final_group =  aggregate_per_month(historical_transactions) 
#１０行表示
gc.collect()
final_group[:10]
train.head()
#trainデータとhistorical_transactions.csv	クレカ取引履歴整形後の
#決済完了フラグが失敗だったデータをカードID(card_id)で左外部結合
#testデータも
train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
train.head()
#trainデータとhistorical_transactions.csv	クレカ取引履歴整形後の
#決済完了フラグが完了だったデータをカードID(card_id)で左外部結合
#testデータも
train = pd.merge(train, authorized, on='card_id', how='left')
test = pd.merge(test, authorized, on='card_id', how='left')
train.head()
#trainデータとnew_merchant_transactions.csv(販売側の新規取引履歴)を左外部結合
#testデータも
train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')
train.head()
#trainデータに上記で算出したfinal_groupを左外部結合
train = pd.merge(train, final_group, on='card_id', how='left')
test = pd.merge(test, final_group, on='card_id', how='left')

#ガベージコレクション
gc.collect()
train.head()
#trainデータに
#クレカ取引履歴(historical_transactions.csv)の
#authorized_flag=カード決済完了フラグの合計と平均をagg_fun変数に格納
#上記と左外部結合
train = pd.merge(train, auth_mean, on='card_id', how='left')
test = pd.merge(test, auth_mean, on='card_id', how='left')
train.head()
#trainデータとtestデータの件数を出してる　201917行の139列　123623行の139列
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
gc.collect()
#カードID(card_id)又は初購入月(first_active_month)以外のカラムを引っ張り出してる　
features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
features
#featureで始まる奴だけ抜き出し
categorical_feats = [c for c in features if 'feature_' in c]
categorical_feats
#LGBMのパラメータ設定
#葉っぱの数(値が大きいほうが良いが、大きすぎると詰まる事がある)
param = {'num_leaves': 31,
         #葉っぱの最低数
         'min_data_in_leaf': 25,
         #regression(回帰手法)
         'objective':'regression',
         #max_depth(決定木の深さ)num_leaves(葉っぱの数)　以下は対応表
         #1は２
         #2は4
         #3は8
         #7は128
         #10は1024
#このルールを覚えてしまえば，特に問題なさそうである．例えば"XGBoost"の max_depth=6 の設定と同等にするには，num_leaves=64 と設定すればよい．
         'max_depth': 7,
         #学習度合い(値が小さいほうが良い)
         #勾配降下のような学習重み。num_roundは、実行する学習ステップの数、つまり構築するツリーの数。
         #高いと学習率が上がるが過学習しやすくなる。ラウンド数を２倍し、etaを２で割る。学習に２倍時間がかかるが、モデルは良くなる。
         'learning_rate': 0.01,
         #何かで使う？
         'lambda_l1':0.13,
         #LightGBM = GBDT(Gradient boosting decision tree) + GOSS(Gradient-based One-Side Sampling) + EFB(Exclusive Feature Bundling)
         #三種ある内の一つ。　詳細はhttps://qiita.com/Sa_qiita/items/7aa98c5df4019a7197ffで
         "boosting": "gbdt",
         #各木を作成するときの列におけるサブサンプルの割合　デフォ１
         #過学習している場合はこの値を下げる。
         "feature_fraction":0.85,
         #サブサンプルを生成する際のトレーニングデータの抽出割合。たとえば、0.5に設定すると、
         #XGBoost はデータの半分をランダムに選んで木を成長させることで、オーバーフィッティングを防ぎます。
         #上記の説明からすると８割トレーニングデータにするのかな？
         'bagging_freq':8,
         #使用するオブジェクトの割合を制御するパラメータ。0と1の間の値。
         "bagging_fraction": 0.9 ,
         #スコアの算出方法？ RMSE（Root Mean Square Error）平均平方二乗誤差
         "metric": 'rmse',
         #警告レベルの表示（計算には関係ない）
         "verbosity": -1,
         #ランダムシードに相当する。ここを大きくしてもスコアが安定した方が良い（？）
         "random_state": 2333}

#%%time
#StratifiedKFold (y, k):分割後のデータセット内のラベルの比率を保ったまま、データをk個に分割。と説明があったので
#５分割してデータの各階層をシャッフルしてランダムシードを設定している？（random_stateが来るのは変な気もするが）
folds = KFold(n_splits=5, shuffle=True, random_state=15)
#trainデータの長さの分だけ0のデータを格納する
oof = np.zeros(len(train))
#同じ処理をtestデータにも行う
predictions = np.zeros(len(test))
#この時点の時間を記録しておいてもう１回time.time()で処理時間を図る
start = time.time()
#データフレームの作成
feature_importance_df = pd.DataFrame()
#enumerateはインデックス番号の取得
#split（X、y =なし、groups =なし）
#X ： 配列のような形（n_samples、n_features）
#学習データ。ここで、n_samplesはサンプル数、n_featuresは特徴数です。
#y ： 配列のような形（n_samples、）
#教師あり学習問題のための目標変数
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    #ilocは列行を番号指定  
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)
    #演算の周回数
    num_round = 10000
    #lgbmの設定でcallback関数？を設定している　下記に設定項目の説明を記述しておきますが分からないのもある。。ぐぬぬ
    
    #early_stopping_rounds（int またはNone 、オプション（デフォルト= None ）） - 早期停止を有効にします。
    #検証スコアが向上しなくなるまで、モデルは学習します。
    #検証スコアは、early_stopping_roundsトレーニングを継続するために少なくともラウンドごとに改善する必要があります。
    #少なくとも1つの検証データと1つのメトリックが必要です。複数ある場合は、それらすべてをチェックします。
    #しかし、トレーニングデータは無視されます。best_iteration早期停止ロジックが設定によって有効にされている場合、
    #最高のパフォーマンスを持つ反復のインデックスがフィールドに保存されます
    
    #verbose_eval（ブール値または整数値、オプション（デフォルト= True ）） -
    #少なくとも1つの検証データが必要です。Trueの場合、有効セットの評価メトリックは各ブースティングステージで出力されます。
    #intの場合、すべてのverbose_evalブースティングステージで有効なセットの評価メトリックが出力されます。
    #最後のブースティングステージまたはを使って見つけたブースティングステージearly_stopping_roundsも印刷されます。
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    print(clf)
    #サイキットラーンの予測　１番目の引数は使うデータ　２番目の引数は目的変数　予測するデータ
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    #データフレーム作成
    fold_importance_df = pd.DataFrame()
    #trainのカラム名を格納したのを追加
    fold_importance_df["feature"] = features
    #予測データを格納？
    fold_importance_df["importance"] = clf.feature_importance()
    print(fold_importance_df)
    #foldを一つついか？
    fold_importance_df["fold"] = fold_ + 1
    #最初のデータフレームに追加
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #分からない
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
#結局importanceがなんなのかわかんねぇ。。。。LGBMの演算で出た結果だと思うけど。。。
feature_importance_df
#上記をfeatureで集計して平均出して昇順にインデックスを並べてる
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
cols
#???feature_importance_dfを複製しただけ？？？？
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
best_features
#importanceの高い順から可視化しただけ
plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
print(gc.collect())
lgbparam = {'num_leaves': 31,
            'boosting_type': 'rf',
             'min_data_in_leaf': 25, 
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.005,
             "min_child_samples": 20,
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.2,
             "verbosity": -1,
            #並列処理を行うスレッド数の指定
             "nthread": 4,
             "random_state": 4590}
#RepeatedKFoldを使う
from sklearn.model_selection import RepeatedKFold
#データを５分割して２回繰り返す？
folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4520)
#さっきもやってたなこれ
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))
#処理時間計測用
start = time.time()
#データフレーム作成
feature_importance_df = pd.DataFrame()
#以下も同じなので割愛
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 11000
    clf = lgb.train(lgbparam, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions_lgb += clf.predict(test[features], num_iteration=clf.best_iteration) / (5 * 2)

print("CV score: {:<8.5f}".format(mean_squared_error(oof_lgb, target)**0.5))
#ここも同じ
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
#カードIDが入ったデータフレーム作成
sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df.head()
#ここに予測結果を放り込む
sub_df["target"] = predictions
sub_df.head()
#csv出力
sub_df.to_csv("submit_lgb.csv", index=False)
#さっきと同じ事を後者の演算で出来た予測で格納
sub_df1 = pd.DataFrame({"card_id":test["card_id"].values})
sub_df1["target"] = predictions_lgb
sub_df1.to_csv("submit_lgb1.csv", index=False)
sub_df1.head()
#予測が入ってたデータ
oof
oof_lgb
#２つを結合して昇順に並べてる
train_stack = np.vstack([oof,oof_lgb]).transpose()
test_stack = np.vstack([predictions,predictions_lgb]).transpose()
#データを５分割して繰り返しは１回？
folds = RepeatedKFold(n_splits=5,n_repeats=1,random_state=4520)
#０の配列つくるやつ
oof_stack = np.zeros(train_stack.shape[0])
predictions_stack = np.zeros(test_stack.shape[0])
#同じ事やってる？
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, target)):
    print("fold n°{}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    print("-" * 10 + "Stacking " + str(fold_) + "-" * 10)
#     cb_model = CatBoostRegressor(iterations=3000, learning_rate=0.1, depth=8, l2_leaf_reg=20, bootstrap_type='Bernoulli',  eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
#     cb_model.fit(trn_data, trn_y, eval_set=(val_data, val_y), cat_features=[], use_best_model=True, verbose=True)
    #ここで線形回帰モジュールを引っ張ってきてる？
    clf = BayesianRidge()
    clf.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf.predict(val_data)
    predictions_stack += clf.predict(test_stack) / 5


np.sqrt(mean_squared_error(target.values, oof_stack))
#提出ファイル書き換え
sample_submission = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')
sample_submission['target'] = predictions_stack
sample_submission.to_csv('Bayesian_Ridge_Stacking.csv', index=False)
#Blend1は今回の結果をBlend2は今回の結果を２割　3.695.csvが２割　combining_submission(1)が６割
sample_submission = pd.read_csv('../input/elo-merchant-category-recommendation/sample_submission.csv')
sample1 = pd.read_csv("../input/elo-blending/3.695.csv")
sample2 = pd.read_csv("../input/elo-blending/combining_submission (1).csv")
sample_submission['target'] = predictions * 0.5 + predictions_lgb * 0.5
sample_submission.to_csv("Blend1.csv", index = False)
sample_submission['target'] = sample_submission['target'] * 0.2 + sample1['target'] * 0.2 + sample2['target'] * 0.6
sample_submission.to_csv('Blend2.csv', index=False)
