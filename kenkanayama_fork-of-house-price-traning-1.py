# 以下のnoteを参考に特徴量を作成している。

# https://www.kaggle.com/comartel/house-price-xgboost-starter
import csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import itertools

import datetime

pd.set_option('display.max_columns', None) # 何これ

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import xgboost as xgb

from operator import itemgetter

import time

from sklearn import preprocessing
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv") # read train data

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv") # read test data
print(len(train))

print(len(train.columns))

train.head()
print(len(test))

print(len(test.columns))

test.head()
# トレーニングデータとテストデータを配列にしたものと、合体させたデータフレームを作る。それからisnullの個数も見ておく

# データフレームの合体は、同じカラムであることを確認してから行うようにする。今回は別途確認してしまったので、そのままマージさせている。

tables=[test,train]



train_sp = train['SalePrice']

train_train = train.drop('SalePrice', axis=1)

merge_df = pd.concat([train_train, test])



total_missing = merge_df.isnull().sum()

print(total_missing)
# でisnullでnull値ががっちゃんこdfのレコード数のうち3分の1以上あるカラムは消す。ここで3分の1以上にするか、どうかは自由に。

# テストデータとトレーニングデータから消す

to_delete = total_missing[total_missing > (len(merge_df)//3)]

for table in tables:

    table.drop(to_delete.index.tolist(),axis=1, inplace=True)

# 消えたカラム

to_delete.index.tolist()
# がっちゃんこDFからもカラムを消しておく

merge_df.drop(to_delete.index.tolist(),axis=1, inplace=True)

merge_df.head()
# カラムに対して数値データとそうじゃないデータを判別し、そのカラム取得する。

numerical_features   = merge_df.select_dtypes(include=["float","int","bool"]).columns.values

categorical_features = merge_df.select_dtypes(include=["object"]).columns.values

print(numerical_features)

print(categorical_features)
# 数値が入っている特徴量カラムのnull値は中央値で置き換え、

# 文字が入っている特徴量カラムのnull値は最も多く出てくる文字で置き換える

# 数値を平均ではなく中央値で置き換えたほうが外れ値の影響を受けにくくさせるためです。

# 中央値や頻出ワードを出す際にがっちゃんこDFを使わないでそれぞれのトレーニングデータDF、テストデータDFから各中央値や頻出ワードを出すほうがよいかも？

for table in tables:

    for feature in numerical_features: 

        table[feature].fillna(merge_df[feature].median(), inplace = True)

    for feature in categorical_features:

        table[feature].fillna(merge_df[feature].value_counts().idxmax(), inplace = True)

# がっちょんこDFも同じく。

for feature in numerical_features: 

    merge_df[feature].fillna(merge_df[feature].median(), inplace = True)

for feature in categorical_features:

    merge_df[feature].fillna(merge_df[feature].value_counts().idxmax(), inplace = True)
# 文字列は数値に変換するためにラベルエンコーディングにかける

# ラベルエンコーディングとワンホットエンコーディングの2つを覚えていれば問題ないと思われる。

for feature in categorical_features:

    le = preprocessing.LabelEncoder()

    le.fit(merge_df[feature])

    for table in tables:

        table[feature] = le.transform(table[feature])
# テストデータ

tables[0].head()
# 学習データ

tables[1].head()
#def get_features(train, test):

trainval = list(tables[1].columns.values)

testval = list(tables[0].columns.values)



# 共通のカラムを取得して、Id列は削除する。

features = list(set(trainval) & set(testval)) # check wich features are in common (remove the outcome column)

features.remove('Id') # remove non-usefull id column
# 以下のURLを見るとよい

# https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-linear-booster-booster-gblinear

# シンプルな xgboost

train = tables[1]

test = tables[0]

target='SalePrice'



eta_list = [0.1,0.2,0.3] # デフォルトは0.3。次のブースティングプロセスに使用する重みをここで決定する。ブースティングプロセスについてはxgboost、決定木について調べればわかる

max_depth_list = [3,4,6,8] # 決定木の深さを3パターン用意してみる

subsample = 0.8 # デフォは1トレーニングデータの何％をサンプリングするかを決める。今回は80%。1にすると過剰適合になりがち。

colsample_bytree = 0.8 # colsample_bytree、colsample_bylevel、colsample_bynodeの3種類があり、デフォは1。列のサブサンプリング用のパラメーターのファミリー。わからないので何も考えなくていい。



num_boost_round = 400

early_stopping_rounds = 10 # 過学習を防ぐ学習方法。検証スコアの改善が止まるまでトレーニングする。今回は10回前と比較して評価指標が改善していなければ学習を停止させるようにしている。

test_size = 0.2



# start the training

array_score=np.ndarray((len(eta_list)*len(max_depth_list),3)) # この後の多くのパターンのスコアを格納するためだけのもの



i=0

# itertools.product()はかっこ内のリストを直積にして全パターンのループができる。

for eta,max_depth in list(itertools.product(eta_list, max_depth_list)):

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))

    params = {

        "objective": "reg:linear", # どんな回帰を使うか。ロジスティック回帰を使うかみたいなものをここで設定する。デフォはreg：squarederrorで二乗損失を伴う回帰

        "booster" : "gbtree", # デフォはgbtree。これが決定木。他にはgblinearまたはdartがある

        "eval_metric": "rmse", # 回帰の場合はrmse、分類の場合はerror、ランキングの場合はmean average precision

        "eta": eta, # 上で説明している

        "tree_method": 'exact', # デフォはauto。auto、exact、approx、hist、gpu_histが使える。autoを選べばexact、approx、hist、gpu_histから適切だと思われるものを選んでくれる

        "max_depth": max_depth,# 決定木の深さ

        "subsample": subsample, # 上で説明している

        "colsample_bytree": colsample_bytree, # 上で説明している

        "silent": 0, #メッセージ出力について設定できるものらしい。有効な値は、0（サイレント）、1（警告）、2（情報）、3（デバッグ）

        "seed": 0, #乱数シードというものらしい。デフォは0。公式でも説明がなくて、、。

    }



    X_train, X_valid = train_test_split(train, test_size=test_size, random_state = 0)# トレーニングデータの8割を使う。2割は検証用に使う。この各割合で使うレコードはランダムに抽出している。

    #y_train = np.log(X_train[target]) # 底をeとするaの対数。aは今回X_train[target]　よくわからん

    #y_valid = np.log(X_valid[target]) # 底をeとするaの対数。X_valid[target]　よくわからん

    y_train = X_train[target] # 底をeとするaの対数。aは今回X_train[target]　よくわからん

    y_valid = X_valid[target] # 底をeとするaの対数。X_valid[target]　よくわからん

    dtrain = xgb.DMatrix(X_train[features], y_train) # トレーニングデータでXGBoostがモデルに使う際の形になるようにデータを生成

    dvalid = xgb.DMatrix(X_valid[features], y_valid) # 検証用データのXGBoostがモデルに使う際の形になるようにデータを生成



    watchlist = [(dtrain, 'train'), (dvalid, 'eval')] # 評価とprintするときに使う。

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True) # トレーニングする。verbose_eval=Trueでイテレーション毎にmerror, stdを表示する。？



    print("Validating...")

    score = gbm.best_score # アーリーストッピングが発生した場合に使える「gbm.best_score」。最良のスコアを入れている

    print('Last error value: {:.6f}'.format(score))

    array_score[i][0]=eta

    array_score[i][1]=max_depth

    array_score[i][2]=score

    i+=1
df_score=pd.DataFrame(array_score,columns=['eta','max_depth','Score'])

print("df_score : \n", df_score)

importance = gbm.get_fscore() # 各特徴量の重要性をゲットする

importance = sorted(importance.items(), key=itemgetter(1), reverse=True)

print('Importance array: ', importance)

np.save("features_importance",importance) # 後で使うためにセーブできるそう。np.loadでロードできる。「features_importance.npy」というファイルが出力されているはず。

print("Predict test set...")



#↓いよいよ予測

test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit) # アーリーストッピングが有効な場合に「ntree_limit=bst.best_ntree_limit」が使える。検証結果のうち最も結果が良かったモデルで予測を行える。





#return test_prediction, score 
#output = np.exp(test_prediction) # np.logで変換したものを直している。のだと思う。

output = test_prediction # np.logで変換したものを直している。のだと思う。

# テストデータとアウトプットデータの数が同じだよね、っていう確認

print(len(test))

print(len(output))
# 出力する

last_list = []

for i , j in zip(test['Id'], output):

    last_list.append({

        'Id' : i,

        'SalePrice' : j

    })

df=pd.DataFrame(last_list)

now = datetime.datetime.now()

sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

df.to_csv(sub_file , index = False)