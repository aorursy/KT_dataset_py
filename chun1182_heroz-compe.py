# 最終提出はversion.6

# 日付使わないほうが精度としては良かった。

# log入れた特徴と入れない特徴の両方入れたほうがなんかよかった。

'''

基本的な流れとしては

last_reviewの欠損値埋め→年月取り出し

緯度経度の正規化→掛け算割り算列用意

各変数のlogとったものを用意

カテゴリ変数をonehot化

PCA特徴量作成

priceはlog1pを取って使う

Boost,RF(,MLP)の判別機を5-foldで作って全部混ぜる、MLPは混ぜなかった

というもの。全部混ぜる力押しに頼っている感じだ

'''
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# カーネルの初めにこれをインストールしないとエラーが出た

!pip install category_encoders
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv("/kaggle/input/heroz-internal-competition/train.csv")

test = pd.read_csv("/kaggle/input/heroz-internal-competition/test_features.csv")

ans = pd.read_csv("/kaggle/input/heroz-internal-competition/sample_submission.csv")

print("Train shape : ", train.shape)

print("Test shape : ", test.shape)

print("ans shape : ", ans.shape)



train
# 欠損値確認、ちょっとあるみたい

train.isnull().sum()
# 欠損値確認、テストだけ欠損してるようなのはないみたい。

test.isnull().sum()
# 全体確認

train.describe(include='all')
# レコード確認、nameは2時間だと使えないかな

test
# ID使えるか一応見てみたけど関係なさそうだから外す

pd.plotting.scatter_matrix(train[['host_id','price']])
# お金とかは基本logとったほうがよさげ、対数とると正規分布っぽくなることが多い

np.log1p(train['price']).hist(bins=20)
# 欠損してるってことはレビューないってことなのかなと

train['reviews_per_month'] = train['reviews_per_month'].fillna(0)

test['reviews_per_month'] = test['reviews_per_month'].fillna(0)
# 欠損してるってことはレビューないってことだから、適当な日付で埋めよう

train['last_review'] = train['last_review'].fillna('2019-12-31')

test['last_review'] = test['last_review'].fillna('2019-12-31')
# レビューさてた日付とか取り出してみた、引き算数するのは正規化的な

train['last_review'] = pd.to_datetime(train['last_review'])

test['last_review'] = pd.to_datetime(test['last_review'])

train['year'] = train['last_review'].dt.year-2010

test['year'] = test['last_review'].dt.year-2010

train['month'] = train['last_review'].dt.month-1

test['month'] = test['last_review'].dt.month-1
# 緯度経度も正規化っぽくして、かけたのと割ったのを作ってみた、斜めも表現

train['latitude'] = train['latitude']-40

train['longitude'] = train['longitude']+75

test['latitude'] = test['latitude']-40

test['longitude'] = test['longitude']+75

train['waru'] = train['latitude'] / train['longitude']

train['kakeru'] = train['latitude'] * train['longitude']

test['waru'] = test['latitude'] / test['longitude']

test['kakeru'] = test['latitude'] * test['longitude']
# 結構偏ってる値が多かったので、log入れたのを作ってみる、効果はあるような無いような

train['minimum_nights_log'] = np.log1p(train['minimum_nights'])

test['minimum_nights_log'] = np.log1p(test['minimum_nights'])

train['calculated_host_listings_count_log'] = np.log1p(train['calculated_host_listings_count'])

test['calculated_host_listings_count_log'] = np.log1p(test['calculated_host_listings_count'])

train['availability_365_log'] = np.log1p(train['availability_365'])

test['availability_365_log'] = np.log1p(test['availability_365'])

train['number_of_reviews_log'] = np.log1p(train['number_of_reviews'])

test['number_of_reviews_log'] = np.log1p(test['number_of_reviews'])

train['reviews_per_month_log'] = np.log1p(train['reviews_per_month'])

test['reviews_per_month_log'] = np.log1p(test['reviews_per_month'])
# こっちのほうがNN的には使ってくれそうな

np.log1p(train['number_of_reviews']).hist()
# ちょっと確認

train.describe()
# 使うカラムを列挙するために表示させる

train.columns
# categoty_encodersのほうが便利と聞いたので。

# onehotにしたい列と、推論で使う列をリストにして、ceに流す

import category_encoders as ce

target = 'price'



list_cols = ['neighbourhood_group', 'neighbourhood', 'room_type',]

target_columns = ['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'room_type', 

       'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'waru', 'kakeru', 'minimum_nights_log',

       'calculated_host_listings_count_log', 'availability_365_log',

       'number_of_reviews_log', 'reviews_per_month_log', 'year', 'month']

# OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。

ce_ohe = ce.OneHotEncoder(cols=list_cols)

train_onehot = ce_ohe.fit_transform(train[target_columns])
'year', 'month'
# 念のため欠損値確認

train_onehot.isnull().sum()
# テストも同様に

test_onehot = ce_ohe.transform(test[target_columns])

# 特徴量が多いときにPCA入れたら効果出たのがあったので

from sklearn.decomposition import PCA

pca2 = PCA(n_components=5)

pca2_results = pca2.fit_transform(train_onehot)

train_onehot['pca0']=pca2_results[:,0]

train_onehot['pca1']=pca2_results[:,1]

train_onehot['pca2']=pca2_results[:,2]

train_onehot['pca3']=pca2_results[:,3]

train_onehot['pca4']=pca2_results[:,4]

pca2_results = pca2.transform(test_onehot)

test_onehot['pca0']=pca2_results[:,0]

test_onehot['pca1']=pca2_results[:,1]

test_onehot['pca2']=pca2_results[:,2]

test_onehot['pca3']=pca2_results[:,3]

test_onehot['pca4']=pca2_results[:,4]
# 再度確認、pca系の値がなんかおかしくてよくない

test_onehot.describe()
# targetはlog1pで、今回は0円がなかったので1pいらないかもだけど何となく

y_train = np.log1p(train["price"])
# 大きさ見てみる

train_onehot.shape
# k-foldにする。何度も使うのでリスト化しておく

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1)

s = list(kf.split(train_onehot, y_train))
# GBRを使う、xgbとかlightgbmのほうがいいと聞いたけど、まだハイパラ調整する練習してないので。

# 5つに分けて学習してそれぞれをリストに保管しておく。

# mseなんかおかしいので使い方間違ってる気がする

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

gbk_models = []

for train_i, val_i in s:

    gbk = GradientBoostingRegressor()

    gbk.fit(train_onehot.iloc[train_i], y_train.iloc[train_i])

    y_pred = gbk.predict(train_onehot.iloc[val_i])

    acc_gbk = round(mean_squared_error(np.expm1(y_pred), np.expm1(y_train[val_i])))

    print(acc_gbk)

    gbk_models.append(gbk)
# 上記同様、nn系もあったほうがバリエーションとしていいのかなと

from sklearn.neural_network import MLPRegressor

mlp_models = []

for train_i, val_i in s:

    mlp = MLPRegressor(max_iter=100, hidden_layer_sizes=(100,100), 

                    activation='relu',  learning_rate_init=0.01)

    mlp.fit(train_onehot.iloc[train_i], y_train.iloc[train_i])

    y_pred = mlp.predict(train_onehot.iloc[val_i])

    acc_gbk = round(mean_squared_error(np.expm1(y_pred), np.expm1(y_train[val_i])))

    print(acc_gbk)

    mlp_models.append(mlp)
# random forestが今のところ何となく好きなので

from sklearn.ensemble import RandomForestRegressor

rfc_models = []

for train_i, val_i in s:

    rfc = RandomForestRegressor()

    rfc.fit(train_onehot.iloc[train_i], y_train.iloc[train_i])

    y_pred = rfc.predict(train_onehot.iloc[val_i])

    acc_gbk = round(mean_squared_error(np.expm1(y_pred), np.expm1(y_train[val_i])))

    print(acc_gbk)

    rfc_models.append(rfc)
# 各モデルを取り出して推論する。最後に平均。重みとかはつけなかった。

# MLPは良くなかったので混ぜなかった。きちんと正規化しとけばよかった。

# expm1で戻しながら

models = gbk_models + rfc_models

preds = np.array([np.expm1(model.predict(test_onehot)) for model in models])

preds = preds.mean(axis=0)

preds = np.where(preds < 0 , 0, preds)
# priceを推論したものに上書きして提出

ans["price"] = preds

ans.to_csv("heroz_nakai.csv", index=False)
# 形変だけど、PCA系が悪さしてるのかも

np.log1p(ans['price']).hist(bins=20)
# どれがよく使ったかを確認してみる,表示はよく使ったのだけでいいや

fti = gbk_models[0].feature_importances_  

print('Feature Importances:')

for i,feat in enumerate(train_onehot.columns):

    if fti[i]>0.0001:

        print('\t{0:10s} : {1:>12.4f}'.format(feat, fti[i]))