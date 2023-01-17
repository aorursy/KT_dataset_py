#モジュールインストール
import numpy as np
import lightgbm as lgb
import pandas as pd
from kaggle.competitions import twosigmanews
import matplotlib.pyplot as plt
import random
from datetime import datetime, date
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time
#データ取得コマンド
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Done!')
#株価とニュースのデータをそれぞれ振り分けて格納
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
#timeを日付のみに整形
market_train_df['time'] = market_train_df['time'].dt.date
#2010年～のデータだけ保存
market_train_df = market_train_df.loc[market_train_df['time']>=date(2010, 1, 1)]
market_train_df.head()
#平行作業のモジュールのインポート
from multiprocessing import Pool

#create_lag(移動平均の平均最大最小標準偏差作成メソッド)
#メソッド定義カッコの中にデフォルト値いれてる
def create_lag(df_code,n_lag=[3,7,14,],shift_size=1):

    #df_codeの資産コードを重複なしでcodeに入れる
    code = df_code['assetCode'].unique()
    
    #return_features(収益機能)からcol変数に
    for col in return_features:
        #n_lagをwindow変数にループ処理
        for window in n_lag:
            #window変数の値の分、現在から以前の移動平均の値を出して値をずらしている
            rolled = df_code[col].shift(shift_size).rolling(window=window)
            #移動平均の平均
            lag_mean = rolled.mean()
            #最大値
            lag_max = rolled.max()
            #最小値
            lag_min = rolled.min()
            #標準偏差
            lag_std = rolled.std()
            #colの値_lag_windowの値_meanのカラム名でdf_codeに格納される
            df_code['%s_lag_%s_mean'%(col,window)] = lag_mean
            df_code['%s_lag_%s_max'%(col,window)] = lag_max
            df_code['%s_lag_%s_min'%(col,window)] = lag_min
            #標準偏差はやらない？
#             df_code['%s_lag_%s_std'%(col,window)] = lag_std
    #df_codeに空値があったら-1にして返す
    return df_code.fillna(-1)


#generate_lag_features(時系列での特徴抽出)
def generate_lag_features(df,n_lag = [3,7,14]):
    #features(特徴変数)に株価データ全部入れてる（項目全書き出ししている理由は不明）
    features = ['time', 'assetCode', 'assetName', 'volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10', 'universe']
    #assetCode(銘柄コード)の重複削除
    assetCodes = df['assetCode'].unique()
    #表示
    print(assetCodes)
    #all_dfとりあえず作ってる？
    all_df = []
    #df_codesにassetCodeで集計(SQLグループバイ)で格納
    df_codes = df.groupby('assetCode')
    #df_codes(全体データ)からdf_code(一日データ)に時間と銘柄コードと将来予測を先頭に放り込んでいるが
    #不明
    df_codes = [df_code[1][['time','assetCode']+return_features] for df_code in df_codes]
    #df_codesの長さを表示
    print('total %s df'%len(df_codes))
    
    #cpu４個で並列処理開始
    pool = Pool(4)
    #create_lagメソッドにdf_codesの引数を代入して出た結果をall_dfに格納している
    all_df = pool.map(create_lag, df_codes)
    
    #new_dfにall_dfをテーブル結合
    new_df = pd.concat(all_df)
    #return_featuresが空値の行を削除
    new_df.drop(return_features,axis=1,inplace=True)
    #並列処理終了
    pool.close()
    #new_dfを返す
    return new_df
#ここはコメントのみだったので設定変更用の控えでは？

# return_features(収益機能)に終値を入れる
# return_features = ['close']
#　new_dfは時系列での特徴抽出のメソッドに株価訓練データを放り込む　移動平均の設定は当日から前５日に設定
# new_df = generate_lag_features(market_train_df,n_lag = 5)
# 株価訓練データにnew_dfをtime assetCodeで外部結合
# market_train_df = pd.merge(market_train_df,new_df,how='left',on=['time','assetCode'])
#return_features(収益機能)の設定（結局終値だけではない）
return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']
#移動平均の日数設定
n_lag = [3,7,14]
#株価訓練データに時系列の特徴抽出メソッドを3,7,14日の移動平均を放り込む
new_df = generate_lag_features(market_train_df,n_lag=n_lag)
new_df.head()
#株価訓練データに移動平均のデータをtime,assetCodeカラムで外部結合
market_train_df = pd.merge(market_train_df,new_df,how='left',on=['time','assetCode'])
#移動平均がちゃんと追加されているかチェック
print(market_train_df.columns)
market_train_df.head()
#mis_imputeメソッド(空値置換)
def mis_impute(data):
    #カラムを一つずつ引っ張ってくる
    for i in data.columns:
        #型が文字列なら空値をotherに置き換え
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        #型が数字なら空値は平均値に置き換え
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data
#株価訓練データを整形したデータに置換
market_train_df = mis_impute(market_train_df)
#data_prep(株価データを入れたら整形してくれるメソッド)
def data_prep(market_train):
    #銘柄コードの重複削除しながら、行番号を加えつつ、カラム一つずつ引っ張る
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    #銘柄コードを上記で処理(多分行番号をassetCodeTを追加してそこに入れてるだけ)
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    #空値の行を削除
    market_train = market_train.dropna(axis=0)
    return market_train
#上記のメソッドを実行
market_train_df = data_prep(market_train_df)
# # shapeはデータが縦何列あって横何行あるかを表記
print(market_train_df.shape)
#assetCodeTがバラバラになっているのが分かる
market_train_df.tail()
#データを数字に変更するモジュールをインポート
from sklearn.preprocessing import LabelEncoder
#10日後に株価が停滞したか上昇したものだけup変数に放り込む
up = market_train_df['returnsOpenNextMktres10'] >= 0

#universe(これが１以外は訓練データとして使えない)の値をuniverse変数に放り込む
#でもなんで値？
universe = market_train_df['universe'].values

universe
#d変数に時間放り込む
d = market_train_df['time']
#最終的に訓練データとして使用するカラムを選択
#カラムを一つずつ引っ張り出してきてはいるが
#下記のカラムは除外する様にしている
fcol = [c for c in market_train_df if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'time_x','provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'universe','sourceTimestamp']]
fcol
#上記で最終的に選択したカラムの値を放り込む
X = market_train_df[fcol].values
X
#returnsOpenNextMktres10のTrueFalseの値を放り込む
up = up.values
up
#rはTrueFalseじゃなくて実際の値を放り込んでいる
r = market_train_df.returnsOpenNextMktres10.values
r
# X値の範囲　(この範囲を後々保持しておくと良いと書いてます)
#それぞれのカラム毎の最小値
mins = np.min(X, axis=0)
mins
#それぞれのカラムの最大値
maxs = np.max(X, axis=0)
maxs
#最大値ー最小値で範囲算出
rng = maxs - mins
rng
#最低値からどれ位の位置に存在しているかをXに挿入
#平均値だったら0.5 上位10%だったら0.9
X = 1 - ((maxs - X) / rng)
X
# 値の整合性チェック 簡単に
assert X.shape[0] == up.shape[0] == r.shape[0]
print(X.shape[0])
print(up.shape[0])
print(r.shape[0])
#それぞれのリストの数が一致している事を確認
#ここからXgboost入れてる
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time
# X_train, X_test, up_train, up_test, r_train, r_test,u_train,u_test,d_train,d_test = model_selection.train_test_split(X, up, r,universe,d, test_size=0.25, random_state=99)
#訓練データとテストデータを分ける作業
#2015年以後のデータはTrue、以前はFalseで処理
#te = market_train_df['time']>date(2015, 1, 1)
#te
#今後のコメントアウトをこのversionでは使わずに
#サイキットラーンのtrain_test_splitを使って訓練データ75%テストデータ25%でそれぞれ放り込んでいる
# X = 生データ 
# up = returnsOpenNextMktres10の正負TrueFalse 
#　r = returnsOpenNextMktres10の値
# u = universeの値　１のデータだけ使う為
# d = 日付
X_train, X_test, up_train, up_test, r_train, r_test,u_train,u_test,d_train,d_test = \
model_selection.train_test_split(X, up, r,universe,d, test_size=0.25, random_state=99)
#2015年以後のデータは2946738件数中2054539件だという事が分かるだけ
#tt = 0
#for tt,i in enumerate(te.values):
#    if i:
#        idx = tt
#        print(i,tt)
#        break
#print(idx)
#訓練データとテストデータに分ける実際の作業(スライジング)
#生データ
#X_train, X_test = X[:idx],X[idx:]
#returnsOpenNextMktres10の正負TrueFalse
#up_train, up_test = up[:idx],up[idx:]
#returnsOpenNextMktres10の値
#r_train, r_test = r[:idx],r[idx:]
#１のデータだけ使う為
#u_train,u_test = universe[:idx],universe[idx:]
#日付
#d_train,d_test = d[:idx],d[idx:]

#LightGBMの訓練データの設定
train_data = lgb.Dataset(X, label=up.astype(int))
#LightGBMのテストデータの設定
test_data = lgb.Dataset(X_test, label=up_test.astype(int))
# LightBGMの機械学習の設定　param_1 param_2に使ってます
x_1 = [0.19000424246380565, 2452, 212, 328, 202]
x_2 = [0.19016805202090095, 2583, 213, 312, 220]
print(up_train)

####################わかりませーん ってかメソッド定義しているのに使ってない

def exp_loss(p,y):
    #yにグラフの凡例を入れてる？
    y = y.get_label()
#     p = p.get_label()
    #expは対数関数だけど分かりません
    grad = -y*(1.0-1.0/(1.0+np.exp(-y*p)))
    hess = -(np.exp(y*p)*(y*p-1)-1)/((np.exp(y*p)+1)**2)
    
    return grad,hess
###########################################
#実際の機械学習のパラメーターチューニング
params_1 = {
        #精度をあげる為の設定も一緒に書いておきます
        #テンプレ
        'task': 'train',
        #テンプレ
        'boosting_type': 'gbdt',
        #regression(分類)だと思われるがなぜかbinary
        'objective': 'binary',
#         'objective': 'regression',
        #学習度合い(値が小さいほうが良い)
        'learning_rate': x_1[0],
        #葉っぱの数(値が大きいほうが良いが、大きすぎると詰まる事がある)
        'num_leaves': x_1[1],
        #葉っぱの最低数
        'min_data_in_leaf': x_1[2],
#         'num_iteration': x_1[3],
        #反復回数(値が大きいほうが良い)
        'num_iteration': 239,
        #この値を大きくすれば精度が上がるが、処理が重くなる。意味は不明
        'max_bin': x_1[4],
        #冗長性の警告の設定(0はエラー出力で1はログ出力、２はデバッグ)
        'verbose': 1
    }
#２個目も同様の手法で別の値に設定
params_2 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
#         'objective': 'regression',
        'learning_rate': x_2[0],
        'num_leaves': x_2[1],
        'min_data_in_leaf': x_2[2],
#         'num_iteration': x_2[3],
        'num_iteration': 172,
        'max_bin': x_2[4],
        'verbose': 1
    }

gbm_1 = lgb.train(params_1,
        train_data,
#大したこと書いてあるわけじゃないのですが、num_boost_roundは少なくとも200でやって、テストデータのerror rateの一番少ない回数選びなさいよとのことです。
#このnum_boost_roundについては、納得出来てない部分もあるので、後ほど少し考察してみます。
#とりあえずで、やってみました。133回が良さそうです。
        #多分学習回数の設定（何週するかとか）
        num_boost_round=133,
        #答え合わせ用データの場所？
        valid_sets=test_data,
        #最低学習回数
        early_stopping_rounds=5,
#         fobj=exp_loss,
        )
#下記も同様に
gbm_2 = lgb.train(params_2,
        train_data,
        num_boost_round=133,
        valid_sets=test_data,
        early_stopping_rounds=5,
#         fobj=exp_loss,
        )
#1個目のデータと2個目のデータの平均出してる
confidence_test = (gbm_1.predict(X_test) + gbm_2.predict(X_test))/2
confidence_test
#予測値の%　割合の算出
confidence_test = (confidence_test-confidence_test.min())/(confidence_test.max()-confidence_test.min())
confidence_test
#%に二乗して-1????
confidence_test = confidence_test*2-1
print(max(confidence_test),min(confidence_test))
confidence_test
# 最終スコアの計算に使用される実際のメトリックの計算
r_test = r_test.clip(-1,1) # -1～１以外の値を取り除く　彼らはどこから来たのかという
#学習の推測地と予測前の目的変数とユニバース値をかけてる？
x_t_i = confidence_test * r_test * u_test
#日付とスコア値だけのデータフレーム作り
data = {'day' : d_test, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
#日付でグループバイ（集約）　で多次元配列を１次元に直してる
x_t = df.groupby('day').sum().values.flatten()
#スコアの平均値
mean = np.mean(x_t)
#スコアの標準偏差
std = np.std(x_t)
#変動係数の逆数
score_test = mean / std
print(score_test)
import gc
del X_train,X_test
gc.collect()
#prediction
days = env.get_prediction_days()
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
total_market_obs_df = []
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if (n_days%50==0):
        print(n_days,end=' ')
    t = time.time()
    market_obs_df['time'] = market_obs_df['time'].dt.date
    
    return_features = ['returnsClosePrevMktres10','returnsClosePrevRaw10','open','close']
    total_market_obs_df.append(market_obs_df)
    if len(total_market_obs_df)==1:
        history_df = total_market_obs_df[0]
    else:
        history_df = pd.concat(total_market_obs_df[-(np.max(n_lag)+1):])
    print(history_df)
    
    new_df = generate_lag_features(history_df,n_lag=[3,7,14])
    market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])
    
#     return_features = ['open']
#     new_df = generate_lag_features(market_obs_df,n_lag=[3,7,14])
#     market_obs_df = pd.merge(market_obs_df,new_df,how='left',on=['time','assetCode'])
    
    market_obs_df = mis_impute(market_obs_df)
    
    market_obs_df = data_prep(market_obs_df)
    
#     market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = (gbm_1.predict(X_live) + gbm_2.predict(X_live))/2
    prediction_time += time.time() -t
    
    t = time.time()
    
    confidence = lp
    confidence = (confidence-confidence.min())/(confidence.max()-confidence.min())
    confidence = confidence * 2 - 1
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    
env.write_submission_file()
sub  = pd.read_csv("submission.csv")
