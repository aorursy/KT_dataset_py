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
## このプログラムは「Digit Recognizer」

## において作成したノートブックにおいて動作します



import numpy as np

import pandas as pd

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import KFold



# train.csvを読み込んでpandasのDataFrameに格納。

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# trainから画像データを抽出してDataFrameオブジェクトに格納。

train_x = train.drop(['label'], axis=1)

# trainから正解ラベルを抽出してSeriesオブジェクトに格納。

train_y = train['label'] 

# test.csvを読み込んでpandasのDataFrameに格納。

test_x = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



# trainのデータを学習データとテストデータに分ける。

kf = KFold(n_splits=4, shuffle=True, random_state=71)

tr_idx, va_idx = list(kf.split(train_x))[0]

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



# 画像のピクセル値を255.0で割って0～1.0の範囲にしてnumpy.arrayに変換。

tr_x, va_x = np.array(tr_x / 255.0), np.array(va_x / 255.0)



# 画像データを2階テンソルを

# (高さ = 28px, 幅 = 28px , チャンネル = 1)の

# 3階テンソルに変換。

# グレースケールのためチャンネルは1。

tr_x = tr_x.reshape(-1,28,28,1)

va_x = va_x.reshape(-1,28,28,1)



# 正解ラベルをOne-Hot表現に変換。

tr_y = to_categorical(tr_y, 10) # numpy.ndarrayオブジェクト

va_y = to_categorical(va_y, 10) # numpy.ndarrayオブジェクト



# x_train、y_train、x_testの形状を出力。

print(tr_x.shape)

print(tr_y.shape)

print(va_x.shape)

print(va_y.shape)
# 畳み込みニューラルネットワーク



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,Dense, Flatten



model = Sequential()                 # Sequentialオブジェクトの生成





# 第1層

model.add(

    Conv2D(filters=32,               # フィルターの数

           kernel_size=(5, 5),       # 5×5のフィルターを使用

           padding='same',           # ゼロパディングを行う

           input_shape=(28, 28, 1),  # 入力データの形状         

           activation='relu'         # 活性化関数はReLU

           ))



# Flatten層

model.add(Flatten())



# 出力層

model.add(Dense(10,                  # 出力層のニューロン数は10

                activation='softmax' # 活性化関数はsoftmax

               ))

    

# オブジェクトのコンパイル

model.compile(

    loss='categorical_crossentropy', # 損失の基準は交差エントロピー誤差

    optimizer='rmsprop',             # オプティマイザーはRMSprop

    metrics=['accuracy'])            # 学習評価として正解率を指定



# モデルの構造を出力。

model.summary()
# 学習を行って結果を出力

history = model.fit(

    tr_x,             # 訓練データ

    tr_y,             # 正解ラベル

    epochs=30,        # 学習を繰り返す回数

    batch_size=100,   # ミニバッチの数

    verbose=1,        # 学習の進捗状況を出力する

    validation_data=(

    va_x, va_y        # 検証用データの指定

    ))
test_x = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



test_x= np.array(test_x)



test_x = test_x.reshape(-1,28,28,1)



result=model.predict(test_x)



#print(result[:5])



y_test = [x.argmax() for x in result]

print(y_test[:10])



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