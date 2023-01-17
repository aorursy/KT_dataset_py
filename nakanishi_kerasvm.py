import keras
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

# relu: 活性化関数。０の時はずっと０、０からプラスは線形の右肩上がりのグラフ

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(10, activation='softmax'))
model.summary()
from keras.models import Model

from keras.layers import Input, Dense
inputs = Input(shape=(784, ))

x = Dense(512, activation='relu')(inputs)

x = Dense(512, activation='relu')(inputs)

x = Dense(512, activation='relu')(inputs)

outputs = Dense(10, activation='softmax')(x)
# どこがInputでどこがOutputかを教える必要あり

model = Model(input=inputs, output=outputs)
# ==== デフォルト設定で使う場合 ====

# optimizerに全て小文字のストリングを渡すだけ（超カンタン！）

# 例えば以下のような設定が使える

# optimizer='sgd'

# optimizer='adadelta'

# optimizer='adam'

# optimizer='rmsprop



# ==== 好みに合わせて設定を変えたい場合 ====

# まず、最適化手法のクラスを読み込む

from keras.optimizers import SGD, Adadelta, Adam, RMSprop

# それらをインスタンス化してからoptimizer引数に渡す

# 例）　SGD with Nesterov momentum を用いる場合

sgd_nesterov=SGD(lr=0.01, momentum=0.9, nesterov=True)



# モデルのコンパイル

model.compile(loss='categorical_crossentropy', # 損失関数（この量のパラメータ勾配で学習する）

              optimizer='adadelta', # 最適化手法（デフォルト設定）

              #optimizer='rmsprop', # 最適化手法（デフォルト設定）

              #optimizer=sgd_nesterov, # 最適化手法（お好み設定）

              metrics=['accuracy'] # 評価指標

             )
from keras.datasets import mnist
# 手書き文字データセット（MNIST）の読み込み

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train
X_train.shape
y_train[:5]
# 可視化用ライブラリの読み込み

%matplotlib inline

import matplotlib.pyplot as plt
# 入力データを可視化（最初の５文字）

fig, ax = plt.subplots(1, 5)



for ii in range(5):

    ax[ii].imshow(X_train[ii], cmap='gray')

    ax[ii].axis('off')
X_train = X_train.reshape(60000, 784)

X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32') / 255.

X_test = X_test.astype('float32') / 255.

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')
from keras.utils import np_utils
nb_classes = 10
# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_train[:5]
y_train[:5]
import time
# TensorBoardで学習の進捗状況をみる

tb_cb = keras.callbacks.TensorBoard(log_dir='/tmp/keras_mnist_mlp', histogram_freq=1)



# バリデーションロスが下がれば、エポックごとにモデルを保存

cp_cb = keras.callbacks.ModelCheckpoint(filepath='./mnist_mlp_best_weight.hdf5', 

                                        monitor='val_loss', verbose=1, save_best_only=True, mode='auto')



# バリデーションロスが５エポック連続で上がったら、ランを打ち切る

es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')



cbks = [tb_cb, cp_cb, es_cb]
tic = time.time()



# 学習を実行

# 学習途中の損失関数の値などはhistory.historyに保存される。

history = model.fit(X_train, Y_train,

                    batch_size=128, 

                    nb_epoch=20,

                    verbose=0,

                    validation_data=(X_test, Y_test)

                    )



toc = time.time()



# 学習にかかった時間を表示

print("Execution time: {0:.2f} [sec]".format(toc - tic))
# テストデータに対する評価値

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0]) # 損失関数の値

print('Test accuracy:', score[1]) # 精度
# 学習曲線

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].set_title('Training performance (Loss)')

ax[0].plot(history.epoch, history.history['loss'], label='loss')

ax[0].plot(history.epoch, history.history['val_loss'], label='val_loss')

ax[0].set(xlabel='Epoch', ylabel='Loss')

ax[0].legend()



ax[1].set_title('Training performance (Accuracy)')

ax[1].plot(history.epoch, history.history['acc'], label='acc')

ax[1].plot(history.epoch, history.history['val_acc'], label='val_acc')

ax[1].set(xlabel='Epoch', ylabel='Accuracy')

ax[1].legend(loc='best')
# 予測値

Y_test_pred = model.predict(X_test)
# 予測の形

Y_test_pred.shape
# 予測の可視化

plt.imshow(Y_test_pred[:10], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
# 入力データを可視化（最初の10文字）

fig, ax = plt.subplots(1, 10, figsize=(10, 2))



for ii in range(10):

    ax[ii].imshow(X_test[ii].reshape(28, 28), cmap='gray')

    ax[ii].axis('off')
plt.imshow(Y_test_pred[:40], cmap='gray', interpolation='nearest', vmin=0, vmax=1)