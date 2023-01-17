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
# トレーニングデータのファイル名一覧を取得

filenames = os.listdir("/kaggle/input/dogs-vs-cats/train/train")

categories = []  # 正解ラベルを格納するリスト



# 取得したファイル名の数だけ処理を繰り返す

for filename in filenames:

    # ファイル名から文字列を切り取る

    category = filename.split('.')[0]

    # 切り取った文字列にdogが含まれていれば'1'

    # そうでなければ'0'をcategoriesに格納

    whichCategories = '1' if category == 'dog' else '0'

    categories.append(whichCategories)

    

# 教師データのDataFrameを作成

df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
print(len(os.listdir("/kaggle/input/dogs-vs-cats/train/train")))

print(len(os.listdir("/kaggle/input/dogs-vs-cats/test/test")))
df.head()
import matplotlib.pyplot as plt

import random

import time

from keras import layers

from keras.layers import Dense, Dropout, GlobalMaxPooling2D, Flatten

from keras.preprocessing.image import load_img

from keras.applications import VGG16

from keras.models import Model, Sequential

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

%matplotlib inline



plt.figure(figsize=(12,12))



TRAIN_DATA = "/kaggle/input/dogs-vs-cats/train/train"



# 9枚の画像を表示してみる

for i in range(9):

    plt.subplot(3, 3, i+1)

    # データからランダムに画像を読み込む

    image = load_img(TRAIN_DATA+'/'+random.choice(df.filename))

    plt.imshow(image)

plt.tight_layout()

plt.show()
image_size = 224

input_shape = (image_size, image_size, 3)
epochs = 7  # エポック数

batch_size = 16  # バッチサイズ
VGG16model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
for layer in VGG16model.layers[:15]:

    layer.trainable = False


last_layer = VGG16model.get_layer('block5_pool')

last_output = last_layer.output



# 512ノードのプーリング層を追加

new_last_layers = GlobalMaxPooling2D()(last_output)

# 512ノードの全結合層を追加、活性化関数はReLU

new_last_layers = Dense(512, activation='relu')(new_last_layers)

# 過学習防止のためドロップアウトを追加、レートは0.5

new_last_layers = Dropout(0.5)(new_last_layers)

# 最後に犬猫を示す2ノードの出力層を追加、活性化関数はシグモイド関数

new_last_layers = layers.Dense(2, activation='sigmoid')(new_last_layers)
# VGG16に定義したblockAの部分を組み込む

model = Model(VGG16model.input, new_last_layers)

# モデルのコンパイル

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])

# サマリーの表示

# 最後の出力層が2ノードになっていることを確認

model.summary()
train_df, validate_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index()

validate_df = validate_df.reset_index()

total_train = train_df.shape[0]

total_validate = validate_df.shape[0]



# 画像加工による画像の水増し定義

train_datagen = ImageDataGenerator(

  # ここでは回転や拡大、反転等、画像加工に係る

  # 各種パラメータを設定している

  rotation_range=15,

  rescale=1./255,

  shear_range=0.2,

  zoom_range=0.2,

  horizontal_flip=True,

  fill_mode='nearest',

  width_shift_range=0.1,

  height_shift_range=0.1

)

# 学習データのジェネレータを作成

train_generator = train_datagen.flow_from_dataframe(

  train_df,

  TRAIN_DATA,

  x_col='filename',

  y_col='category',

  class_mode='categorical',

  target_size=(image_size, image_size),

  batch_size=batch_size

)

# 検証データのジェネレータ作成

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

  validate_df,

  TRAIN_DATA,

  x_col='filename',

  y_col='category',

  class_mode='categorical',

  target_size=(image_size, image_size),

  batch_size=batch_size

)
history = model.fit_generator(

  train_generator,  # 学習データのジェネレータ

  epochs=epochs,  # エポック数

  # 検証データのジェネレータ

  validation_data=validation_generator,

  validation_steps=total_validate//batch_size,

  steps_per_epoch=total_train//batch_size)
TEST_DATA = "/kaggle/input/dogs-vs-cats/test/test"



filenames = os.listdir(TEST_DATA)

sample = random.choice(filenames)

img = load_img(TEST_DATA+'/'+sample, target_size=(224,224))

plt.imshow(img)

img = np.asarray(img)

img = np.expand_dims(img, axis=0)



predict = model.predict(img)  # 犬か猫か分類

dog_vs_cat = np.argmax(predict, axis=1)

print('dog') if dog_vs_cat==1 else print('cat')
test_filenames = os.listdir(TEST_DATA)

test_df = pd.DataFrame({

    'filename' : test_filenames

})

nb_samples = test_df.shape[0]



# テストデータのジェネレータを作成

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

  test_df,

  TEST_DATA,

  x_col='filename',

  y_col=None,

  class_mode=None,

  batch_size=batch_size,

  target_size=(image_size, image_size),

  shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
dog_vs_cat = np.argmax(predict, axis=1)

submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = dog_vs_cat

submission_df.drop(['filename'], axis=1, inplace=True)

# ファイルに出力

submission_df.to_csv('submission.csv', index=False)
f = open('submission.csv')

print(f.read())

f.close()
img = load_img(TEST_DATA+'/5713.jpg')

plt.imshow(img)