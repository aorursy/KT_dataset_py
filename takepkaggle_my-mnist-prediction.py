# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

#train # 1+28x28 列
test = pd.read_csv("../input/test.csv")

#test # 28x28 列
from keras.models import Sequential

from keras.utils import np_utils, plot_model

from keras.layers import Dense, Activation, Dropout, LSTM

from keras.initializers import glorot_uniform, orthogonal, TruncatedNormal

from keras.callbacks import EarlyStopping

from keras.layers.recurrent import GRU, SimpleRNN



from keras.optimizers import SGD

from keras.datasets import mnist

from keras.utils import np_utils



import math #数値計算

import itertools #順列・組み合わせ

import time



import matplotlib.pyplot as plt #グラフ

import os



from sklearn.model_selection import train_test_split
# 分類数（0から9の10）

num_classes = 10

# 画像の高さと幅

img_height, img_width = 28, 28



def create_model():

    model = Sequential() # オブジェクトの作製

    

    # 畳み込み層

    # Conv2D()

    model = add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)))

    # 畳み込み層

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

    # プーリング層（情報量の削減）

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Dropout(0.25)) # ドロップアウト

    

    # 畳み込み層

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    # 畳み込み層

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    # プーリング層（情報量の削減）

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten()) # 多層パーセプトロンに入力するにあたって4次元配列を1次元配列に変換

    model.add(Dense(512, activation='relu')) # 全結合層

    model.add(Dropout(0.5)) # ドロップアウト

    model.add(Dense(num_classes)) # 全結合層

    model.add(Activation('softmax')) # 活性化層

    

    return model
np.random.seed(1671) # 乱数

NB_EPOCH = 200 # エポック数

BATCH_SIZE = 128 # バッチサイズ

VERBOSE = 1 # kerasのログの詳細さ（0は非表示、1ならプログレスバーを表示、2で全て表示）

NB_CLASSES = 10 # 分類数？

OPTIMIZER = SGD() # 確率的勾配降下法

N_HIDDEN = 128 # 隠れ層のユニット数

VALIDATION_SPLIT = 0.2 # 何割を検証データとして分割するか
# 訓練データを読み込む

df_train = pd.read_csv('../input/train.csv')

df_X_train = df_train.drop('label', axis=1) # ラベル列を削除

df_y_train = df_train['label'] # ラベル列を抽出



# テストデータを読み込む

df_X_test = pd.read_csv('../input/test.csv')
print('df_X_train', df_X_train.shape)

print('df_X_train', type(df_X_train))

print('\ndf_y_train', df_y_train.shape)

print('df_y_train', type(df_y_train))

print('\ndf_X_test', df_X_test.shape)

print('df_X_test', type(df_X_test))



X_train = df_X_train.values

y_train = df_y_train.values

X_test = df_X_test.values



print('\n↓pandasデータフレームからnumpy配列に変換↓\n')

print('X_train', X_train.shape)

print('X_train', type(X_train))

print('\ny_train', y_train.shape)

print('y_train', type(y_train))

print('\nX_test', X_test.shape)

print('X_test', type(X_test))
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')
X_train /= 255

X_test /= 255



#print(X_train.shape[0], 'train samples')

#print(X_test.shape[0], 'test samples')



# カテゴリ変数に変換

#Y_train = np_utils.to_categorical(y_train, NB_CLASSES)

#Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)



print(y_train)

print('\n↓カテゴリ変数の生成↓\n')

print(Y_train)
RESHAPED = 784



# モデル生成

model = Sequential()

model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,))) # 全結合層

model.add(Activation('softmax')) # 活性化関数

model.summary() # モデルの要約を出力
"""

def make_tensorboard(set_dir_name=''):

    tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())

    directory_name = tictoc

    log_dir = set_dir_name + '_' + directory_name

    os.mkdir(log_dir)

    tensorboard = TensorBoard(log_dir=log_dir)

    return tensorboard



callbacks = [make_tensorboard(set_dir_name='keras_MNIST_V1')]

"""
# モデルをコンパイル

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])



model.fit(X_train, Y_train, 

          batch_size=BATCH_SIZE,

          epochs=NB_EPOCH,

          verbose=VERBOSE,

          validation_split=VALIDATION_SPLIT)
answer = model.predict(X_test, verbose=VERBOSE)
df_answer = pd.DataFrame(answer) # pandasデータフレームに変換

#df_answer.max(axis=1)

df_answer = df_answer.idxmax(axis=1) # 最大値の列名を取得

df_answer.index = df_answer.index + 1 # インデックスを1から振りなおす

df_answer.index.names = ['ImageId'] # インデックス名を変更

#print(type(df_answer))

df_answer.name = 'Label' # 列名を変更

#print(df_answer)

df_answer.to_csv('answer.csv', header=True, index=True)



df_answer = pd.read_csv('answer.csv')

print(df_answer)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print('Test score:', score[0])

print('Test accuracy:', score[1])