%matplotlib inline

from keras.models import Sequential, load_model

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers import LeakyReLU

from keras.callbacks import ModelCheckpoint,History,EarlyStopping,LearningRateScheduler

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.optimizers import Adam, Adadelta, RMSprop

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

print(data.shape)
test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

print(test_data.shape)
train = data[:]

val = data[55000:]

train_label = np.float32(train.label)

val_label = np.float32(val.label)

train_image = np.float32(train[train.columns[1:]])

val_image = np.float32(val[val.columns[1:]])

test_image = np.float32(test_data[test_data.columns[1:]])

print('train shape: %s'%str(train.shape))

print('val shape: %s'%str(val.shape))

print('train_label shape: %s'%str(train_label.shape))

print('val_label shape: %s'%str(val_label.shape))

print('train_image shape: %s'%str(train_image.shape))

print('val_image shape: %s'%str(val_image.shape))

print('test_image shape: %s'%str(test_image.shape))
datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range = 10,

    horizontal_flip = False,

    zoom_range = 0.15)
# one-hot coding

from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder(sparse=False,categories='auto')

yy = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

encoder.fit(yy)

# transform

train_label = train_label.reshape(-1,1)

val_label = val_label.reshape(-1,1)



train_label = encoder.transform(train_label)

val_label = encoder.transform(val_label)



print('train_label shape: %s'%str(train_label.shape))

print('val_label shape: %s'%str(val_label.shape))
plt.imshow(train_image[13].reshape(28,28))

plt.show()

print(train_image[13].shape)



train_image = train_image/255.0

val_image = val_image/255.0

test_image = test_image/255.0



train_image = train_image.reshape(train_image.shape[0],28,28,1)

val_image = val_image.reshape(val_image.shape[0],28,28,1)

test_image = test_image.reshape(test_image.shape[0],28,28,1)

print('train_image shape: %s'%str(train_image.shape))



print('train_image shape: %s'%str(train_image.shape))

print('val_image shape: %s'%str(val_image.shape))
model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1),padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(128, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(256, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(256,kernel_regularizer=regularizers.l2(0.02)))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.3))

# model.add(Dense(256,kernel_regularizer=regularizers.l2(0.02)))

# model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

# model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))



model.summary()
BATCH_SIZE = 256

EPOCHS = 53
model.compile(loss='categorical_crossentropy',optimizer=Adadelta(),metrics=['accuracy'])

# fit data

datagen.fit(train_image)



# training

history = model.fit_generator(datagen.flow(train_image,train_label, batch_size=BATCH_SIZE),

                              epochs = EPOCHS,

                              shuffle=True,

                              validation_data = (val_image,val_label),

                              verbose = 1,

                              steps_per_epoch=train_image.shape[0] // BATCH_SIZE)
# 绘制训练 & 验证的准确率值

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# 绘制训练 & 验证的损失值

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
test_label = model.predict(test_image)

# label = np.argmax(label_hot,1)

# id_ = np.arange(0,label.shape[0])
label_hot = model.predict(test_image)

label = np.argmax(label_hot,1)

id_ = np.arange(0,label.shape[0])
sim = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

print(sim.head(10))
save = pd.DataFrame({'id':id_,'label':label})

print(save.head(10))

save.to_csv('submission.csv',index=False)