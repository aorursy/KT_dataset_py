import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2 as cv # (pip install opencv-python)
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers.pooling import GlobalAveragePooling2D

%matplotlib inline
# Train/test data path. 学習・検証用データが格納されているパス
path = '../input/traffic-signs-classification/myData'
# Get classes from sub directory's name. サブディレクトリの名前が交通標識のクラス名（分類名）になっているので、サブディレクトリのリストを取得
classes = os.listdir(path)
print(f'Total number of categories （クラス分類数）: {len(classes)}')

# A dictionary which contains class and number of images in that class. クラス分類ごとのデータ数を取得して格納
counts = {}
for c in classes:
    counts[c] = len(os.listdir(os.path.join(path, c)))

total = sum(list(counts.values()))
print(f'Total number of images in dataset （全データ数）: {total}')
# Number of images in each clsss plot. 学習・検証用画像データのラベルとその件数をグラフ表示
fig = plt.figure(figsize = (25, 5))
sns.barplot(x = list(counts.keys()), y = list(counts.values())).set_title('Number of train/test images in each class')
plt.xticks(rotation = 90)
plt.margins(x=0)
plt.show()
# Image size. 学習・検証用データの画像サイズ
# ベースとするモデルの種類によっては、最小の画像サイズが異なるようなので注意する
# たとえば Xception は 71x71 以上。ResNet50 は 197x197 以上ということだが、実は 32x32 でも大丈夫？（ https://keras.io/ja/applications/ ）
image_width = 32 # 32, 64, 96, 128
image_height = 32 # 32, 64, 96, 128

# The images are RGB.
image_channels = 3

# Get image and label data with ImageDataGenerator
datagen = ImageDataGenerator()
data = datagen.flow_from_directory(path,
                                    target_size=(image_width, image_height),
                                    batch_size=total,
                                    class_mode='categorical',
                                    shuffle=True )

X , y = data.next()

# Labels are one hot encoded
print(f"Data Shape 画像データのリストの構造:{X.shape}")
print(f"Labels shape 正解ラベルのリストの構造:{y.shape}")
fig, axes = plt.subplots(10,10, figsize=(18,18))
for i,ax in enumerate(axes.flat):
    r = np.random.randint(X.shape[0])
    ax.imshow(X[r].astype('uint8'))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Label: '+str(np.argmax(y[r])))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
print("Train Shape（学習用データ）: {}\nTest Shape（検証用データ）: {}".format(X_train.shape, X_test.shape))
# Export model file name. 学習済みモデルの保存ファイル名
best_model_file_path = './best_model.hdf5'

# ModelCheckpoint. 複数回学習を繰り返す中で、一番精度がよかったパターンのモデルを保存する
checkpoint = ModelCheckpoint(best_model_file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# EarlyStopping.　過学習を防止するための仕組み。学習の精度が上がらなくなった段階で実行を停止する
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='max', restore_best_weights=True)

# ReduceLROnPlateau. 精度の改善が停滞した時に学習率を減らす
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
#model = Sequential()
#
#model.add(Conv2D(64, (3, 3), padding='same',
#                 input_shape=X_train.shape[1:]))
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(128, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(128, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#
#model.add(Flatten())
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(y_test.shape[1]))
#model.add(Activation('softmax'))
## initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#
## Let's train the model using RMSprop
#model.compile(loss='categorical_crossentropy',
#              optimizer=opt,
#              metrics=['accuracy'])
#
#loss_history = []
#model.summary()
#for i in range(10):
#    loss_history += [model.fit(X_train, y_train,
#                               validation_data=(X_test, y_test), 
#                               batch_size = 256,
#                               epochs = 1, shuffle="batch")]
# ベースとするモデルの中の一部のレイヤーのみを再学習対象にする場合（末尾の10層のみを再学習対象にする、など）は 'imagenet' を指定、
# ベースとするモデルの全ての層で再学習させる場合は、None を指定
weight = 'imagenet' # None

#base_model = ResNet50(weights=weight, include_top=False, input_shape=(image_width,image_height,image_channels))
#base_model = Xception(weights=weight, include_top=False, input_shape=(image_width,image_height,image_channels))
base_model = MobileNet(weights=weight, include_top=False, input_shape=(image_width,image_height,image_channels))

top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Dropout(0.5)(top_model) # 入力にドロップアウトを適用する． 訓練時の更新においてランダムに入力ユニットを0とする割合であり，過学習の防止に役立ちます．
predictions = Dense(len(classes), activation='softmax')(top_model)
model = Model(inputs=base_model.input, outputs=predictions)

# Set layer to trainable / non-trainable. ベースとするモデルのどの層までを固定にして、どの層からを再学習対象にするかを決定
# たとえば Xception モデルは全部で 135層のレイヤーから構成されているので、そのうちの130層目までを、固定、それ以降を再学習対象とする、という設定ができる
# （ MobileNet は 90層、ResNet50 は 178層）
if weight == 'imagenet':
    trainable = False
    layer_index_freezed = 85
    index = 0
    for layer in model.layers:
        if index > layer_index_freezed:
            trainable = True
    
        layer.trainable = trainable

        # ただし、それ以前の層でも、Batch Normalization 層は再学習させる（参考： https://qiita.com/mokoenator/items/6d7b8f670d3d1250d516 ）
        # Batch Normalization とは： http://tozensou32.blog76.fc2.com/blog-entry-40.html など
        if layer.name.endswith('bn'):
            layer.trainable = True
        
        index = index + 1

# Model layers. モデルの層の数を表示
print("Model layers: {}層".format(len(model.layers)))
# Model summary. モデルの構造を表示
print("Model summary: モデルの構造：")
model.summary()

# Optimizer. オプティマイザ（最適化アルゴリズム）
# adam、adamax、sgd などが指定できる（ https://keras.io/ja/optimizers/ ）
optimizer = 'adam' # 'adamax', 'sgd'

# Compile the model. モデルを構築
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Epochs. エポック数
# 多いほど学習の回数が増えるが、多すぎると過学習となる可能性も高まる
epochs = 5 # 5, 10, 20, 30, ...

# Batch size. バッチサイズ
batch_size = 32 # 32, 64, 128, ...

# Fit the model. 学習を実行
history = model.fit(x=X_train, 
                    y=y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=1, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, checkpoint, reduce_lr])
# Plot learning curves
fig = plt.figure(figsize = (17, 4))
    
plt.subplot(121)
plt.plot(history.history['accuracy'], label = 'acc')
plt.plot(history.history['val_accuracy'], label = 'val_acc')
#plt.plot(history.history['acc'], label = 'acc')
#plt.plot(history.history['val_acc'], label = 'val_acc')
plt.legend()
plt.grid()
plt.title(f'accuracy')

plt.subplot(122)
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.grid()
plt.title(f'loss')
# Loading weights from best model. 一番精度の良かったパターンの結果を読み込む
model.load_weights(best_model_file_path)

# Evaluation result. 性能評価
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
fig, axes = plt.subplots(5,5, figsize=(18,18))
for i,ax in enumerate(axes.flat):
    r = np.random.randint(X_test.shape[0])
    ax.imshow(X_test[r].astype('uint8'))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Original: {} Predicted: {}'.format(np.argmax(y_test[r]), np.argmax(model.predict(X_test[r].reshape(1, image_width, image_height, image_channels)))))