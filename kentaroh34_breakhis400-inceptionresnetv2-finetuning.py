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
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
import os
#import Pillow
#from PIL import Image
#import opencv
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, initializers

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np




#num_train = 1000              # 訓練データの画像数
#num_validation = 1000          # テストデータの画像数

img_h, img_w = 400,400
channels = 3
batch_size = 30               # ミニバッチのサイズ
train_dir = '/kaggle/input/breakhis-400x/BreaKHis 400X/train' # 訓練データのフォルダー

result_dir = '/kaggle/output/Kaggle/working'        # VGG16の出力結果を保存するフォルダー
test_dir = '/kaggle/input/breakhis-400x/BreaKHis 400X/test'

result_dir='/kaggle/ouput/kaggle/working'
train_normal_files = os.listdir(os.path.join(train_dir,'benign'))
print(len(train_normal_files))
#371

train_malignant_files = os.listdir(os.path.join(train_dir,'malignant'))
print(len(train_malignant_files))
#777

test_normal_files = os.listdir(os.path.join(test_dir,'benign'))
print(len(test_normal_files))
#371

test_malignant_files = os.listdir(os.path.join(test_dir,'malignant'))
print(len(test_malignant_files))
#777


import cv2

# !ls -lh '/content/gdrive/My Drive/Colab Notebooks/BreakHis/data/train'

normal_files = os.listdir(os.path.join(train_dir,'benign'))[:12]
print(normal_files)

pneumonia_files = os.listdir(os.path.join(train_dir,'malignant'))[:12]
print(pneumonia_files)


import numpy as np
import cv2
from matplotlib import pyplot as plt

%matplotlib inline

fig = plt.figure(figsize=(12,20))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

for i in range(12):
  img = cv2.imread(os.path.join(train_dir,'benign',normal_files[i]))
  ax = fig.add_subplot(6,2,i+1)
  ax.set_title('benign'+'- '+normal_files[i])
  a='('+str(img.shape[0])+','+str(img.shape[1])+')'
  ax.set_xlabel(a)
  ax.imshow(img)
plt.show()

fig = plt.figure(figsize=(12,20))
plt.subplots_adjust(wspace=0.5, hspace=0.5)

for i in range(12):
  img = cv2.imread(os.path.join(train_dir,'malignant',pneumonia_files[i]))
  ax = fig.add_subplot(6,2,i+1)
  ax.set_title('malignant'+'- '+normal_files[i])
  a='('+str(img.shape[0])+','+str(img.shape[1])+')'
  ax.set_xlabel(a)
  ax.imshow(img)
plt.show()

from keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
import os
#import Pillow
#from PIL import Image
#import opencv
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, initializers

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
#モデル作成・学習
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
import tensorflow as tf
import keras
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D,Input
from keras.layers import Dense, Dropout, Flatten, Activation,GlobalAveragePooling2D,Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras import optimizers

img_h, img_w = 400,400
channels = 3
batch_size = 30  
nb_classes = 1

# InceptionResNetV2のロード。FC層は不要なので include_top=False
input_tensor = Input(shape=(img_h, img_w, 3))
inception_v3 = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)

inception_v3.summary()
## InceptionResNetV2モデルと学習済み重みを読み込む

img_h, img_w = 400,400
channels = 3
batch_size = 30  
nb_classes = 1

# InceptionResNetV2のロード。FC層は不要なので include_top=False
input_tensor = Input(shape=(img_h, img_w, 3))
inception_v3 = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)




# InceptionResNetV2の一部の重みを固定（frozen）
for layer in inception_v3.layers[:775]:
    layer.trainable = False




# FC層の作成
model = Sequential()

model.add(inception_v3)

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', name='sigmoid'))

model.summary()
## モデルのコンパイル
# 最適化はRMSpropで行う
# 学習率を小さくしたのはファインチューニングを

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
img_h, img_w = 400,400
channels = 3
batch_size = 30               # ミニバッチのサイズ
#train_data_dir = 'content/data/train' # 訓練データのフォルダー
#validation_data_dir = 'content/data/validation' # テストデータのフォルダー
#result_dir = 'results'        # VGG16の出力結果を保存するフォルダー  


## 訓練データを読み込むジェネレーターを生成
# データ拡張を行う
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,      # 40度の範囲でランダムに回転させる
    width_shift_range=0.2,  # 横サイズの0.2の割合でランダムに水平移動
    height_shift_range=0.2, # 縦サイズの0.2の割合でランダムに垂直移動
    horizontal_flip=True,   # 水平方向にランダムに反転、左右の入れ替え
    vertical_flip=True,
    zoom_range=0.8,         # ランダムに拡大
    shear_range=0.2         # シアー変換をかける
)

# 訓練データを生成するするジェネレーター
train_generator = train_datagen.flow_from_directory(
    train_dir,             # 訓練データのフォルダー
    target_size=(img_h, img_w), # 画像をリサイズ
    batch_size=batch_size,      # ミニバッチのサイズ
    class_mode='binary'         # 出力層は二値のラベルが必要
)


## テストデータを読み込むジェネレーターを生成
test_datagen = ImageDataGenerator(rescale=1.0 / 255)


# テストデータを生成するするジェネレーター
validation_generator = test_datagen.flow_from_directory(
    test_dir,        # テストデータのフォルダー
    target_size=(img_h, img_w), # 画像をリサイズ
    batch_size=batch_size,      # ミニバッチのサイズ
    class_mode='binary'         # 出力層は二値のラベルが必要
)
%%timeit

epochs=5
num_train = 1148
num_validation = 545

## 学習を行う

# 訓練データのジェネレーターのサイズ：39
print(len(train_generator))
# 訓練データの数をミニバッチのサイズで割った値：38
print(num_train//batch_size)
# テストデータのジェネレーターのサイズ：19
print(len(validation_generator))
# テストデータの数をミニバッチのサイズで割った値：18
print(num_validation//batch_size)

# モデルのファインチューニング
history = model.fit(
    
    # 訓練データのジェネレーター
    train_generator,
    # 各エポックにおけるステップ数として
    # 訓練データの数をミニバッチのサイズで割った値を指定
    steps_per_epoch=num_train//batch_size,
    # エポック数（学習回数）
    epochs=epochs,
    # テストデータのジェネレーター
    validation_data=validation_generator,
    # テストにおける各エポックにおけるステップ数として
    # テストデータの数をミニバッチのサイズで割った値を指定
    validation_steps=num_validation//batch_size
)


# model save
model.save(os.path.join(result_dir,'breakhis_InceptionResNetV2.h5'))

import pandas as pd

df_history=pd.DataFrame(history.history)

df_history.to_pickle(os.path.join(result_dir,'InceptionResNetV2_history.pkl'))
%matplotlib inline
import matplotlib.pyplot as plt

def plot_acc_loss(history):
    # 精度の推移をプロット
    plt.plot(history.history['accuracy'],"-",label="accuracy")
    plt.plot(history.history['val_accuracy'],"-",label="val_acc")
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の推移をプロット
    plt.plot(history.history['loss'],"-",label="loss",)
    plt.plot(history.history['val_loss'],"-",label="val_loss")
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
    
# 損失と精度をグラフに出力
plot_acc_loss(history)