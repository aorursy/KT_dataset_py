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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train
import numpy as np

import pandas as pd

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import KFold





# train.csvを読み込んでpandasのDataFrameに格納。

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# trainから画像データを抽出してDataFrameオブジェクトに格納。

train_x = train.drop(['label'], axis=1)

train_x

# trainから正解ラベルを抽出してSeriesオブジェクトに格納。

train_y = train['label'] 
# trainのデータを4分割し、訓練用に3、バリデーション用に1の割合で配分する。

kf = KFold(n_splits=4, shuffle=True, random_state=123)

kf

type(kf)

# 訓練用とバリデーション用のレコードのインデックス配列を取得。

tr_idx, va_idx = list(kf.split(train_x))[0]

tr_idx[:20]

va_idx[:20]
va_idx[:20]

print(len(tr_idx))

print(type(tr_idx))

print(len(va_idx))

a=np.array([2,3,4])

print(a)

print(type(a))
# 訓練用とバリデーション用のレコードのインデックス配列を取得。

tr_idx, va_idx = list(kf.split(train_x))[0]



# 訓練とバリデーション用の画像データと正解ラベルをそれぞれ取得。

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



# 画像のピクセル値を255.0で割って0～1.0の範囲にしてnumpy.arrayに変換。

tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)



# 正解ラベルをOne-Hot表現に変換。

tr_y = to_categorical(tr_y, 10) # numpy.ndarrayオブジェクト

va_y = to_categorical(va_y, 10) # numpy.ndarrayオブジェクト



# x_train、y_train、x_testの形状を出力。

print(tr_x.shape)

print(tr_y.shape)

print(va_x.shape)

print(va_y.shape)

from collections import Counter



# 0～9の各数字の枚数を調べる。

count = Counter(train['label'])

count
import seaborn as sns



# 0～9の各数字の枚数をグラフにする。

sns.countplot(train['label'])
# 訓練データの1番目の要素を出力。

print(tr_x[0])

print(type(tr_x))

print(tr_x.shape)

print(tr_x[0].shape)
import matplotlib.pyplot as plt

%matplotlib inline



# 訓練データから50枚抽出してプロットする。

plt.figure(figsize=(12,10))

x, y = 10, 5 # 10列5行で出力。

for i in range(50):  

    plt.subplot(y, x, i+1)

    # 28×28にリサイズして描画する。

    plt.imshow(tr_x[i].reshape((28,28)),interpolation='nearest')

plt.show()
tr_x.shape[1]
# ニューラルネットワークの構築



# keras.modelsからSequentialをインポート

from tensorflow.keras.models import Sequential

# keras.layersからDense、Activationをインポート

from tensorflow.keras.layers import Dense, Activation



# Sequentialオブジェクトを生成。

model = Sequential()



# 第1層(隠れ層)

model.add(Dense(

    128,                     # ユニット数は128。

    input_dim=tr_x.shape[1], # 入力データの形状を指定。

    activation='sigmoid'     # 活性化関数はSigmoid。

))



# 第2層(出力層)

model.add(Dense(

    10,                  # ニューロン数はクラスの数と同数の10。

    activation='softmax' # マルチクラス分類に適したSoftmaxを指定。

))



model.compile(

    # 損失関数はクロスエントロピー誤差関数。

    loss='categorical_crossentropy',

    # オプティマイザーはAdam。

    optimizer='adam',

    # 学習評価として正解率を指定。

    metrics=['accuracy'])



# モデルの構造を出力。

model.summary()
# 学習を行う。

result = model.fit(tr_x, tr_y,                   # 訓練データと正解ラベル。

                   epochs=5,                     # 学習回数を5回にする。

                   batch_size=100,               # ミニバッチのサイズは100。

                   validation_data=(va_x, va_y), # 検証用のデータを指定。

                   verbose=1)     
# test.csvを読み込んでpandasのDataFrameに格納。

test_x = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_x
# テストデータで予測して結果をNumPy配列に代入する。

result = model.predict(test_x)

print(result.shape)
for x in result[:5]:

    print(x.argmax())
[x.argmax() for x in result[:5]]
# 予測した数字をNumPy配列に代入する。

y_test = [x.argmax() for x in result]

y_test
# 提出用のCSVファイルをデータフレームに読み込む。

submit_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

# 先頭から5行目までを出力。

submit_df.head()
# データフレームのLabel行に予測値を格納する。

submit_df['Label'] = y_test

# 先頭から5行目までを出力。

submit_df.head()
# データフレームの内容を提出用のCSVファイルに書き込む。

import os

print(os.getcwd())



submit_df.to_csv('submission.csv', index=False)