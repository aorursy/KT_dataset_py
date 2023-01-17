%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

# データを読み込む
trains = pd.read_csv('../input/train.csv')
submit_images = pd.read_csv('../input/test.csv')
# train,test,valの三つに分割
train, test = train_test_split(trains, test_size = 0.05)
train, val = train_test_split(train, test_size = 0.1)

# ラベルと画像に分ける
train_labels = train['label']
train_images = train.drop(labels = ['label'], axis=1)
val_labels = val['label']
val_images = val.drop(labels = ['label'], axis=1)
test_labels = test['label']
test_images = test.drop(labels = ['label'], axis=1)

# 画像の前処理
train_images = train_images.values.reshape((-1, 28, 28, 1))
train_images = train_images.astype('float32') / 255
val_images = val_images.values.reshape((-1, 28, 28, 1))
val_images = val_images.astype('float32') / 255
test_images = test_images.values.reshape((-1, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# submit画像も同じように前処理 
submit_images = submit_images.values.reshape((-1, 28, 28, 1))
submit_images = submit_images.astype('float32') / 255

# labelをone-hotに変換
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=8,
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

# trainデータのみ拡張することに注意
datagen.fit(train_images)
# 学習率を変化させる
callbacks_list = [
    ReduceLROnPlateau(
    monitor='val_acc',
    factor=0.1,
    patience=5,
    verbose=1
    )
]
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

batch_size = 128
history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size = batch_size),
                              epochs = 50,
                              callbacks = callbacks_list,
                              validation_data = (val_images, val_labels),
                              steps_per_epoch = train_images.shape[0] // batch_size)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# 正解率をプロット
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylim(0.975, 1.000)
plt.legend()

plt.figure()

# 損失値をプロット
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylim(0.0, 0.1)
plt.legend()

plt.show()
model.evaluate(test_images, test_labels)
results = model.predict(submit_images)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)