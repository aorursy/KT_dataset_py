# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

train = pd.read_csv(f'../input/train-test/train.tsv', sep='\t') #sep='\t'でタブ区切り
test = pd.read_csv(f'../input/train-test/test.tsv', sep='\t')
#item_condition_id別に見てみる

train['item_condition_id'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.hist(train.loc[train['item_condition_id'] == 1, 'price'].dropna(),
        range=(0, 250), bins=30, label='1')
plt.ylim(0, 220000)
plt.title('id=1')
plt.xlabel('price')
plt.ylabel('count')
plt.subplot(2, 3, 2)
plt.hist(train.loc[train['item_condition_id'] == 2, 'price'].dropna(),
        range=(0, 250), bins=30, label='2')
plt.ylim(0, 220000)
plt.title('id=2')
plt.subplot(2, 3, 3)
plt.hist(train.loc[train['item_condition_id'] == 3, 'price'].dropna(),
        range=(0, 250), bins=30, label='3')
plt.ylim(0, 220000)
plt.title('id=3')
plt.subplot(2, 3, 4)
plt.hist(train.loc[train['item_condition_id'] == 4, 'price'].dropna(),
        range=(0, 250), bins=30, label='4')
plt.ylim(0, 220000)
plt.title('id=4')
plt.subplot(2, 3, 5)
plt.hist(train.loc[train['item_condition_id'] == 5, 'price'].dropna(),
        range=(0, 250), bins=30, label='5')
plt.ylim(0, 220000)
plt.title('id=5')
print(train.loc[train['item_condition_id'] == 1, 'price'].mean())
print(train.loc[train['item_condition_id'] == 2, 'price'].mean())
print(train.loc[train['item_condition_id'] == 3, 'price'].mean())
print(train.loc[train['item_condition_id'] == 4, 'price'].mean())
print(train.loc[train['item_condition_id'] == 5, 'price'].mean())
#箱ひげ図描画
import numpy as np

sns.set()
price_1 = [np.log1p(w) for w in train.loc[train['item_condition_id'] == 1, 'price']]
price_2 = [np.log1p(w) for w in train.loc[train['item_condition_id'] == 2, 'price']]
price_3 = [np.log1p(w) for w in train.loc[train['item_condition_id'] == 3, 'price']]
price_4 = [np.log1p(w) for w in train.loc[train['item_condition_id'] == 4, 'price']]
price_5 = [np.log1p(w) for w in train.loc[train['item_condition_id'] == 5, 'price']]
points = (price_1, price_2, price_3, price_4, price_5)
plt.boxplot(points)
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel('item_condition_id')
plt.ylabel('log(price+1)')
from contextlib import contextmanager
import time

#contextmanagerはwith文の実行時間を計測する
#@はデコレータ
#print f は文字列内に変数を直接渡せる
#xfは小数点第x位までを出力する

"""with文の実行時間を返すデコレータ"""
@contextmanager
def timer(name):
    t0 = time.time()
    yield 
    print(f'[name] done in {time.time() - t0:.0f} s')
import pandas as pd

#: -> は型指定(アノテーション)
#fillna(任意の文字列)で欠損値を置換

"""相関のあるカラムを結合"""
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    #nameカラムはnameとbrand_nameカラムで結合
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'] + fillna('')
    #textカラムを追加し,item_descriptionとnameとcategory_nameで結合
    df['text'] = df['item_description'].fillna('') + ' ' + df['name'].fillna('') + ' ' + df['category_name'].fillna('')
    return df[['name', 'text', 'shipping', 'item_condition_id']]
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from operator import itemgetter

#位置引数の*は引数をタプル化する(ポインタではない!).
#FunctionTransformer : itemgetter()は文字列を抽出する.validate=Trueでnumpy配列に変換.
#make_pipelineはPipilineの省略形.推定量の名前が推定量の指定タイプの小文字になる.
#>>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
#    Pipeline(steps=[('standardscaler', StandardScaler()),
                #('gaussiannb', GaussianNB())])

"""独自メソッドのPipeline作成"""
def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=True), *vec)
from typing import List, Dict

#オプションにorient='records'を指定すると、行ごとの辞書を保持したJSONになる

"""pd.DataFrame型オブジェクトを辞書型に変換する"""
def to_records(df : pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')
import tensorflow as tf

a = tf.constant(1.5)
b = tf.constant(2.0)
c = tf.constant(-1.0)
d = tf.constant(3.0)

x = tf.add(a, b)
y = tf.multiply(c, d)
z = tf.add(x, y)

tf.print([x, y, z])
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline 

import math


from subprocess import check_output
#ディレクトリ参照
print(check_output(["ls", "../input"]).decode("utf8"))
#assert文:単体テスト

"""RMSLEの定義"""
def RMSLE(y, y_pred):
    assert len(y)==len(y_pred), "Length of y and y_pred are not equal."
    to_sum = math.sqrt(sum([(math.log(y_pred[i] + 1) - math.log(y[i] + 1))**2 for i in range(len(y))])*(1.0/len(y)))
    return to_sum
train.info()
#各カラムの欠損値をmissingで置換

def handle_missing(dataset):
    dataset.category_name.fillna(value='missing', inplace=True)
    dataset.brand_name.fillna(value='missing', inplace=True)
    dataset.item_description.fillna(value='missing', inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)
train.head()
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

#np.hstack()は列方向にデータを結合(行方向はnp.vstack)

"""category_nameを数値データに変換"""
le = LabelEncoder()
le.fit(np.hstack([train.category_name, test.category_name]))
train.category_name = le.transform(train.category_name)
test.category_name = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train.brand_name = le.transform(train.brand_name)
test.brand_name = le.transform(test.brand_name)
del le
train.head()
from keras.preprocessing.text import Tokenizer

#nameとitem_descriptionを列方向でつなげる
raw_text = np.hstack([train.name.str.lower(), train.item_description.str.lower()])

#kerasにおける前処理:テキストをtoken単位に分割
tok_raw = Tokenizer()
#fit_on_textsメソッドで単語と数値(ベクトル)を対応づける(数値に変換はまだしない)
#テキストを単語単位に分割する必要もなく,すごく便利
tok_raw.fit_on_texts(raw_text)

#テキストデータを数列に変換
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())

train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
train.head()
#SEQUENCES VARIABLES ANALYSIS

#nameの単語数の最大値
max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), np.max(test.seq_name.apply(lambda x: len(x)))])
#item_descriptionの最大値
max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x)))
                                   , np.max(test.seq_item_description.apply(lambda x: len(x)))])

print("max name seq "+str(max_name_seq))
print("max item desc seq "+str(max_seq_item_description))
train.seq_name.apply(lambda x: len(x)).hist()
plt.xlabel('seq_name\'s length')
plt.ylabel('seq_name\'s word counts')
train.seq_item_description.apply(lambda x: len(x)).hist()
plt.xlabel('seq_item_description\'s length')
plt.ylabel('seq_item_description\'s word counts')
#EMBEDDINGS MAX VALUE
#Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 10 #本当は17だが,ヒストグラムから件数が少ないので10とした.
MAX_ITEM_DESC_SEQ = 75
#seq_nameとseq_item_descriptionの中で最長のテキストの単語数を抽出
MAX_TEXT = np.max([np.max(train.seq_name.max())
                   , np.max(test.seq_name.max())
                  , np.max(train.seq_item_description.max())
                  , np.max(test.seq_item_description.max())])+2

#categoryの最大値
MAX_CATEGORY = np.max([train.category_name.max(), test.category_name.max()])+1
#brand_nameの最大値
MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()])+1
#item_condiion_idの最大値
MAX_CONDITION = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1
from sklearn.preprocessing import MinMaxScaler

#SCALE target variable
train['target'] = np.log(train.price+1)
#MinMaxScalerは変換後のデータの最大・最小を独自に変換・設定する
target_scaler = MinMaxScaler(feature_range=(-1, 1))
#データ変換の施行
array = np.empty(len(train.target)) #maximum supported dimension for an ndarray is 32, found 1482535 の回避
array[:] = train.target             ##maximum supported dimension for an ndarray is 32, found 1482535 の回避
train['target'] = target_scaler.fit_transform(array.reshape(-1,1)) #-1から1の範囲で標準化
pd.DataFrame(train.target).hist()
plt.xlabel('log(price + 1)')
plt.ylabel('count')
#学習データを訓練と検証用に分ける
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
print(dtrain.shape)
print(dvalid.shape)
"""kerasによる学習用のデータセットの定義"""
from keras.preprocessing.sequence import pad_sequences

#pad_sequencesはリストのリストの各要素に対してmaxlenで指定した長さのシーケンスを返す.
#pad_sequencesの戻り値: shapeが(len(sequences), maxlen)のNumpy配列．

#kerasでの学習用にデータを整形(dict(キー, value=Numpy形式))
def get_keras_data(dataset):
    X = {
        'name' : pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc' : pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name' : np.array(dataset.brand_name),
        'category_name' : np.array(dataset.category_name),
        'item_condition' : np.array(dataset.item_condition_id),
        'num_vars' : np.array(dataset[["shipping"]])
    }
    return X

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)
"""準備"""

print(X_train["item_condition"].shape)
print(X_train["name"].shape[1])
print(X_train["item_condition"].shape)
#kerasモデルの定義(nn)
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

"""コールバック関数の定義(訓練中にモデル内部の状態と統計量を可視化する際に使う)"""
def get_callbacks(filepath, patience=2):
    #earlystopping:2エポック数間で改善が見られない場合処理を終了
    es = EarlyStopping('val_loss', patience=patience, mode='min')
    #各エポック終了後にモデルを保存
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

"""誤差関数"""
#これは学習内における損失関数として定義
def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

"""モデルの定義"""
def get_model():
    """ニューラルネットワーク作成"""
    #params
    #ドロップアウトする割合
    dr_r = 0.1
    
    #kerasテンソルのインスタンス化
    #Inputs
    #shape引数は期待される入力次元の値
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    #カテゴリーはもともと階層構造になっていたから、それを分けて別々の入力として予測してみたい.
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #正の整数(インデックス)を固定次元の密ベクトルに変換する
    #Embedding(入力次元数の上限, 出力次元数)
    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    #rnn layers
    #GRUで時系列データ処理
    #nameとitem_descriptionは文章で,単語の並びに意味があるから時系列データとして処理する
    #活性化関数(activation=)を指定しなければtanhが使われる
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)
    
    #main layer
    #入力層
    #concatenateは平滑化した同じshapeの複数のテンソルをまとめてひとつのテンソルにする(引数はリストととして渡す)
    main_layer = concatenate([
                Flatten()(emb_brand_name), #brand_name
                Flatten()(emb_category_name), #category_name
                Flatten()(emb_item_condition), #item_condition_id
                rnn_layer1, #item_description
                rnn_layer2, #name
                num_vars #shipping
                ])
    
    #隠れ層
    #Dropoutを適用して過学習を防ぎたい
    #Dense:全結合ニューラルネットワーク, actiovation=に何も指定しなければ線形活性化関数が使われる
    main_layer = Dropout(dr_r)(Dense(128) (main_layer))
    main_layer = Dropout(dr_r)(Dense(64) (main_layer))
    
    #output
    #出力層
    output = Dense(1, activation="linear")(main_layer)
    
    #Modelクラスはaを入力としてbを計算する際に必要となるあらゆる層を含む.
    model = Model(inputs=[name, item_desc, brand_name, category_name, item_condition, num_vars], outputs=output)
    """学習モデルの作成"""
    #学習モデルの作成はcompileメソッド
    #metricsにリストを渡すことで複数の評価関数で対照実験できる(mae: Mean Absolute Error 絶対値平均誤差)
    model.compile(loss='mse', optimizer='adam', metrics=["mae", rmsle_cust])
    
    return model    
model = get_model()
model.summary()
"""モデルの学習"""
#ミニバッチのサイズ
BATCH_SIZE = 20000
#エポック数(訓練データをシャッフルして学習し直す回数)
epochs = 5

model = get_model()
#学習
model.fit(X_train, #訓練データ
         dtrain.target, #予測ターゲット: log(price + 1)
         epochs=epochs, #エポック数
         batch_size=BATCH_SIZE, #無作為抽出の数
         validation_data=(X_valid, dvalid.target), #モデル評価用のデータセット(検証データ).学習ではここで宣言されたデータは用いない
         verbose=1 #進行状況の表示モード(0, 1, 2)
         )

#検証データで評価
val_preds = model.predict(X_valid) #学習から除いた検証データで評価(出力はターゲットであるlog(price + 1)の一次元リスト)
val_preds = target_scaler.inverse_transform(val_preds) #標準化から元に戻す
val_preds = np.exp(val_preds) - 1 #log(price + 1)からpriceへの変換

#rmsle
y_true = np.array(dvalid.price.values)
y_pred = val_preds 

v_rmsle = RMSLE(y_true, y_pred)
print('RMSLE ERROR : ' + str(v_rmsle))
submission = test[["test_id"]]
submission
#テストデータで予測
preds = model.predict(X_test, batch_size=BATCH_SIZE) #テストデータでlog(price+1)を予測
preds = target_scaler.inverse_transform(preds) #標準化から元に戻す
preds = np.exp(preds) - 1

submission["price"] = preds
submission.head()
submission.to_csv("./mercari_submission_sub.csv", index=False)