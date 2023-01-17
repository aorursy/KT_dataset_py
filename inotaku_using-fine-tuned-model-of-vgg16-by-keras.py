import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

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
# Data preparation.

# データの確認をします。

print(os.listdir("../input"))
# Check the data.

# トレーニングデータ10件をリストします。

print(os.listdir("../input/train/train")[:10])

# テストデータ10件をリストします。

print(os.listdir("../input/test1/test1")[:10])
# Get category from file name.Because there is no correct 

# answer label, get the label from the file name and make 

# it a classification class.

# First, get a list of file names.

# トレーニングデータのファイル名からdog, catを取得し分類するカテゴリとします。

# トレーニングデータのファイル名一覧を取得。

filenames = os.listdir("../input/train/train")

# Variable to store categories.

# クラスを格納する変数。

categories = []

# Perform processing for the number of acquired files.

# 取得したファイル数分処理を繰り返します。

for filename in filenames:

    # Cut out label from file name.

    # ファイル名から正解ラベルを切り取る。

    category = filename.split('.')[0]

    # If the file name contains Dog, set the class to 1, 

    # otherwise set it to 0.

    # ファイル名にDogが含まれていれば、クラスに1を設定し、

    # そうでない場合は0を設定する。

    whichCategorys = '1' if category == 'dog' else '0'

    # Add label.

    # ラベルを変数に格納します。

    categories.append(whichCategorys)



# Create a data frame with file name and class, 

# and use it as supervised learning data.

# ファイル名とクラスを持つデータフレームを作成し、教師ありの学習データとします。

df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})



# display the beginning.

# 先頭を表示してみます。

df.head()

# We have the same amount of images of dogs and cats.

# 犬猫は同数あることが確認できます。

#df['class'].value_counts()

df['category'].value_counts()
# See sample image.

# 画像を表示してみます。

plt.figure(figsize=(12, 12))

for i in range(0, 9):

    plt.subplot(3, 3, i+1)

    image = load_img('../input/train/train/'+df.filename[i])

    plt.imshow(image)

plt.tight_layout()

plt.show()
# Input data is a color image (3 channels) with an image size of 224x224.(Same as VGG 16 input layer)

# 入力データは、画像サイズを224x224のカラー画像(3チャンネル)とする。(VGG16の入力層に合わせた)

image_size = 224

input_shape = (image_size, image_size, 3)
# エポック数7、バッチサイズを16に設定。

epochs = 7

batch_size = 16
# VGG16 model download(Set up Internet connection from Kernel beforehand)

# The output layer of the VGG 16 is 1,000 classes, and this time the output 

# layer is replaced for 2-class classification. 

# VGG16モデルのダウンロード(事前にKernelからInternet接続ができるよう設定しておきます)

# VGG16は1,000クラスの出力層となっており、今回は犬、猫の２クラス分類となるため出力層の取り替えを行います。

# このためinclude_top=Falseとし、出力層の前の層を利用します。

VGG16model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
# Display model summary.

# モデルのサマリを表示します。

# 入力層と畳み込み層、プーリング層からなるブロックが５つある事がわかります。

VGG16model.summary()
# 最後の畳み込み層の直前までの層は、学習されたパラメータをそのまま利用するため

# 今後のトレーニングによって変更されないようにします。今回最後の層だけ学習させます。

# Freeze the layer just before the last convolutional layer.

for layer in VGG16model.layers[:15]:

    layer.trainable = False
# ５ブロック目の畳み込み層だけ学習できる状態になっている事を確認します。

# Only the 5th block can learn.

for layer in VGG16model.layers[:-1]:

    print(layer.trainable)
# Set the 5th block of the VGG16 model as the last output layer.

# VGG16モデルの5ブロック目を最後の出力層とする。

last_layer = VGG16model.get_layer('block5_pool')

last_output = last_layer.output
# Create a new output layer for 2 class classification.

# 分類は犬猫の２クラスの分類を出力する層を新規に作成する。

# プーリング層を置き、入力はVGG16の出力を受け取るようにする。

new_last_layers = GlobalMaxPooling2D()(last_output)

# Add a fully connected layer with 512 hidden units and ReLU activation

# 512ノードの全結合層を追加、活性化関数はReLU

new_last_layers = Dense(512, activation='relu')(new_last_layers)

# Add a dropout rate of 0.5

# ドロップアウトを追加、レートは0.5

new_last_layers = Dropout(0.5)(new_last_layers)

# Add a final sigmoid layer for classification

# 最後に犬猫のクラスを示すノード２つの出力層を作り、シグモイド関数を適用する

new_last_layers = layers.Dense(2, activation='sigmoid')(new_last_layers)
# Combine the VGG 16 with the output layer.

# VGG16と出力層を結合する。

model = Model(VGG16model.input, new_last_layers)

# complile.

# モデルのコンパイル。

model.compile(loss = "categorical_crossentropy",

              optimizer=SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])

# Display model summary.

# サマリ表示

model.summary()
# Prepare training data and validation data.

# トレーニングデータ(train_df)と検証データ(validate_df)を準備する。

train_df, validate_df = train_test_split(df, test_size=0.1)

# indexのリセット

train_df = train_df.reset_index()

validate_df = validate_df.reset_index()

# データ数の取得

total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

print('Total amount of data={}, Total train={}, Total validate={}'.format(len(df), total_train, total_validate))
train_df.head()
validate_df.head()
# Traning Generator

# トレーニングデータの拡張を行う

train_datagen = ImageDataGenerator(

    # 画像をランダムに回転範囲

    rotation_range=15,

    # 画素値のリスケーリング係数

    rescale=1./255,

    # シアー強度（反時計回りのシアー角度）

    shear_range=0.2,

    # ランダムにズームする範囲

    zoom_range=0.2,

    # 水平方向に画像反転

    horizontal_flip=True,

    # 入力画像の境界周りを埋める指定

    fill_mode='nearest',

    # 水平シフトする範囲

    width_shift_range=0.1,

    # 垂直シフトする範囲

    height_shift_range=0.1

)
# Generate a batch of expanded data from data frames and directory.

# トレーニングデータのジェネレータ

# データフレームとパスからデータを拡張したバッチを生成する

train_generator = train_datagen.flow_from_dataframe(

    # トレーニングデータのデータフレーム

    train_df, 

    # トレーニングデータのパス

    "../input/train/train/",

    # ファイル名

    x_col='filename',

    # 正解ラベル(カテゴリ)

    y_col='category',

    # '犬(1)'、'猫(0)'のカテゴリ分類として扱う

    class_mode='categorical',

    # 対象のデータサイズ

    target_size=(image_size, image_size),

    # バッチサイズ

    batch_size=batch_size

)
# Validation Generator

# 検証データのジェネレータ

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    "../input/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='categorical',

    target_size=(image_size, image_size),

    batch_size=batch_size

)
# Prepare a generator for sample display of extended image

# 拡張画像のサンプル表示を行うジェネレータを準備

example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df, 

    "../input/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='categorical'

)

# Display a sample of expanded image data

# 拡張した画像データのサンプルを表示する

plt.figure(figsize=(12, 12))

for i in range(0, 9):

    plt.subplot(3, 3, i+1)

    for X_batch, Y_batch in example_generator:

        image = X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()

# トレーニング時間測定のためタイムスタンプを取得

start = time.time()



# Fit Model

# fine-tune the model

# トレーニングの実施

history = model.fit_generator(

    train_generator,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size)



# Display of learning time

# トレーニング時間の表示

elapsed_time = time.time() - start

print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# evaluation

# モデルの評価

loss, accuracy = model.evaluate_generator(validation_generator, total_validate//batch_size, workers=12)

print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
# loss and accuracy graph

# 損失関数の値と分類精度のグラフ

def plot_model_history(model_history, acc='acc', val_acc='val_acc'):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(range(1,len(model_history.history[acc])+1),model_history.history[acc])

    axs[0].plot(range(1,len(model_history.history[val_acc])+1),model_history.history[val_acc])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history[acc])+1),len(model_history.history[acc])/10)

    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

    

plot_model_history(history)
# Try image recognition.

# Run this cell several times and try out various images.

# 試しに、ランダムに１枚画像を選んで表示してみましょう。

# このセルは何度か実行し、色々な画像で試してみましょう。

filenames = os.listdir("../input/test1/test1")

sample = random.choice(filenames)

img = load_img("../input/test1/test1/"+sample,target_size=(224,224))

plt.imshow(img)

img = np.asarray(img)

img = np.expand_dims(img, axis=0)



predict =  model.predict(img)

dog_vs_cat= np.argmax(predict,axis=1)

print('The animals in the picture are "', end='')

if dog_vs_cat == 1:

    print('dog".')

else:

    print('cat".')

# Preparation of test data.

# テスト用データの準備。

test_filenames = os.listdir("../input/test1/test1")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
# Create Testing Generator.

# テストジェネレータを作成。

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "../input/test1/test1/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    batch_size=batch_size,

    target_size=(image_size, image_size),

    shuffle=False

)

# Predict

# 予測する

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
# Display some of the prediction results.

# 予測結果の一部を表示。

dog_vs_cat= np.argmax(predict,axis=1)

plt.figure(figsize=(12, 12))

for i in range(0, 9):

    ax= plt.subplot(3, 3, i+1, xticks=[], yticks=[])

    image = load_img('../input/test1/test1/'+test_df.filename[i])

    plt.imshow(image)

    ax.set_title("predict={}".format(('dog' if dog_vs_cat[i]==1 else 'cat')))

plt.tight_layout()

plt.show()
# Submission file output

# サブミッション用のDFを準備

submission_df = test_df.copy()

#idを設定

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

#labelを設定

submission_df['label'] = dog_vs_cat

#filenameは不要なので削除

submission_df.drop(['filename'], axis=1, inplace=True)

# サブミッションファイルの出力

submission_df.to_csv('submission.csv', index=False)