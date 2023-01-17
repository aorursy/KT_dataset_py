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
%matplotlib inline

import matplotlib.pyplot as plt



def draw(X):

    '''描画を行う関数

    

       X: (28, 28, 1)の形状をした画像データのリスト

    '''

    plt.figure(figsize=(8, 8)) # 描画エリアは8×8インチ

    pos = 1                    # 画像の描画位置を保持

    

    # 画像の枚数だけ描画処理を繰り返す

    for i in range(X.shape[0]):

        plt.subplot(5, 5, pos) # 4×4の描画領域のpos番目の位置

        # インデックスiの画像を(28,28)の形状に変換して描画

        plt.imshow(X[i].reshape((28,28)),interpolation='nearest')

        plt.axis('off')        # 軸目盛は非表示

        pos += 1

    plt.show()
# データを用意する



import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from tensorflow.keras.utils import to_categorical



# train.csvを読み込んでpandasのDataFrameに格納

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

tr_x = train.drop(['label'], axis=1) # trainから画像データを抽出

train_y = train['label']             # trainから正解ラベルを抽出



# 画像のピクセル値を255.0で割って0～1.0の範囲にしてnumpy.arrayに変換

tr_x = np.array(tr_x / 255.0)



# 画像データの2階テンソルを

# (高さ = 28px, 幅 = 28px , チャンネル = 1)の

# 3階テンソルに変換

# グレースケールのためチャンネルは1

tr_x = tr_x.reshape(-1,28,28,1)



# 正解ラベルをone-hot表現に変換

tr_y = to_categorical(train_y, 10)



# テストで使用する画像の枚数

batch_size = 25
print(type(tr_x))

print(tr_x.shape)

print(tr_y.shape)
# オリジナルの画像を表示

draw(tr_x[0:batch_size])
# ImageDataGeneratorのインポート

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# 回転処理　最大90度

datagen = ImageDataGenerator(rotation_range=90)

g = datagen.flow(           # バッチサイズの数だけ拡張データを作成

    tr_x, tr_y, 25, shuffle=False)

X_batch, y_batch = g.next() # 拡張データをリストに格納

draw(X_batch)               # 描画

print(X_batch.shape)

print(y_batch.shape)
# 平行移動　最大0.5

datagen = ImageDataGenerator(width_shift_range=0.3)

g = datagen.flow(           # バッチサイズの数だけ拡張データを作成

    tr_x, tr_y, batch_size, shuffle=False)

X_batch, y_batch = g.next() # 拡張データをリストに格納

draw(X_batch)               # 描画
# 垂直移動　最大0.5

datagen = ImageDataGenerator(height_shift_range=0.3)

g = datagen.flow(           # バッチサイズの数だけ拡張データを作成

    tr_x, tr_y, batch_size, shuffle=False)

X_batch, y_batch = g.next() # 拡張データをリストに格納

draw(X_batch)               # 描画
# ランダムに拡大　最大0.5

datagen = ImageDataGenerator(zoom_range=0.3)

g = datagen.flow(           # バッチサイズの数だけ拡張データを作成

    tr_x, tr_y, batch_size, shuffle=False)

X_batch, y_batch = g.next() # 拡張データをリストに格納

draw(X_batch)               # 描画
# 左右をランダムに反転

datagen = ImageDataGenerator(horizontal_flip=True)

g = datagen.flow(           # バッチサイズの数だけ拡張データを作成

    tr_x, tr_y, batch_size, shuffle=False)

X_batch, y_batch = g.next() # 拡張データをリストに格納

draw(X_batch)               # 描画
# 上下をランダムに反転

datagen = ImageDataGenerator(vertical_flip=True)

g = datagen.flow(           # バッチサイズの数だけ拡張データを作成

    tr_x, tr_y, batch_size, shuffle=False)

X_batch, y_batch = g.next() # 拡張データをリストに格納

draw(X_batch)               # 描画