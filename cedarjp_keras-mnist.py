# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keras.models import Sequential  # https://keras.io/ja/getting-started/sequential-model-guide/
from keras.layers import Dense,  Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from keras.optimizers import RMSprop, Adam, Adamax, Nadam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, MaxPool2D
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib inline 
sns.set(style='white', context='notebook', palette='deep')
# データ読み込み
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

Y_train = train["label"]  # ラベルだけ
X_train = train.drop(labels = ["label"], axis = 1)  # ラベルを削除

g = sns.countplot(Y_train)
# [0...255]の値を[0.0...1.0]にする
X_train = X_train / 255.0
test = test / 255.0
# 28x28x1の3Dマトリックスにする
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# 0〜9までのラベルをバイナリのクラス行列に変換
# 例: 2 -> [0,0,1,0,0,0,0,0,0,0]
# https://keras.io/ja/utils/np_utils/
Y_train = to_categorical(Y_train, num_classes = 10)
# 訓練用データと検証用データに分割
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# CNNモデル
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',  activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',  activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# オプティマイザ定義
# https://keras.io/ja/optimizers/#rmsprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# モデルのコンパイル
# 損失関数 https://keras.io/ja/losses/#categorical_crossentropy
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# https://keras.io/ja/callbacks/#reducelronplateau
# 評価値の改善が止まったときに学習率をへらす
# monitor: 監視する値
# patienceで指定したエポック数の間改善がなければ学習率をfactor下げる
# verbose=1 学習率を下げたときにメッセージ表示
# min_lr: 学習率の下限
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 10
batch_size = 86
# https://keras.io/ja/preprocessing/image/#imagedatagenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)