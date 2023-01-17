import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
# データセットの準備
# 入力画像の大きさ(行と列）
img_rows, img_cols = 28, 28

# 学習データとテストデータに分割したデータ
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データを高速かつ効率的に使えるPandasをインポート
import pandas as pd

#正解ラベルの表示
print(y_train[53238])

#正解ラベルに紐づく画像データの表示
# Xの53238番目のデータをtest_numberへ切り出す
test_number = x_train[53238]
# reshape関数を使って784を28x28へ変換する
test_number_image = test_number.reshape(28,28)
# pandasのカラム表示の設定を変更
pd.options.display.max_columns = 28
# Numpy配列からPandasのデータフレームへ変換
number_matrix = pd.DataFrame(test_number_image)
# number_matrixの表示
number_matrix
# imshowを使って表示test_number_image（Numpy配列）を画像で表示
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline 
plt.imshow(test_number_image, cmap = matplotlib.cm.binary,
interpolation='nearest')
plt.show()
# backendがTensorFlowとTheanoで配列のshapeが異なるために2パターン記述
if K.image_data_format() == 'channels_first':
    # 1次元配列に変換
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # 1次元配列に変換
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# Xの53238番目のデータをtest_numberへ切り出す
test_number = x_train[53238]
# reshape関数を使って784を28x28へ変換する
test_number_image = test_number.reshape(28,28)
# pandasのカラム表示の設定を変更
pd.options.display.max_columns = 28
# Numpy配列からPandasのデータフレームへ変換
number_matrix = pd.DataFrame(test_number_image)
# number_matrixの表示
number_matrix
# ラベルをバイナリベクトルとして扱う
# Kerasはラベルを数値ではなく、0or1を要素に持つベクトルで扱うため
"""
例えば、サンプルに対するターゲットが「5」の場合次のような形になります。
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
"""
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

test_number = y_train[53238]
test_number
"""
mnist: 手書き数字画像データセット
Sequential: Kerasを用いてモデルを生成するためのモジュール
Dense: 全結合層のレイヤモジュール
Dropout: ドロップアウトモジュール
Conv2D: 2次元畳み込み層のモジュール
MaxPool2D: 2次元最大プーリング層のモジュール
Flatten: 入力を平滑化するモジュール
"""

batch_size = 128
epochs = 10

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5),
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('tanh'))
model.add(Dropout(0.25))

model.add(Conv2D(16, (5,5)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('tanh'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(120))
model.add(Activation('tanh'))
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 損失関数,最適化関数,評価指標を指定してモデルをコンパイル
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# モデルの学習
hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


score=model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('test acc:', score[1])

#学習のグラフ化

#Accuracy：予測した値と正解が一致していた数のカウント。
#Loss：実際のラベルからどのくらい違っていたのかを考慮できる

epochs = range(1, len(hist.history['accuracy']) + 1)

plt.plot(epochs, hist.history['loss'], label='Training loss', ls='-') #損失値
plt.plot(epochs, hist.history['val_loss'], label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, hist.history['accuracy'],  label='Training acc') #正解率
plt.plot(epochs, hist.history['val_accuracy'], label="Validation acc")
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#重みづけデータの保存
model.save('mnist_cnn_model.h5') 
#予測
x = x_test[1:30]  # 先頭30件で予測させる
y = model.predict(x)
test_y = y_test[1:30] 

x_classes = [np.argmax(v, axis=None, out=None) for v in test_y]
print('正解ラベル: ', x_classes)

# one-hotベクトルで結果が返るので、数値に変換する
y_classes = [np.argmax(v, axis=None, out=None) for v in y]
print('予測結果　: ', y_classes)