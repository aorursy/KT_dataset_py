# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedShuffleSplit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

BASE_DIR = '/kaggle/input/Kannada-MNIST'
df_train = pd.read_csv(f'{BASE_DIR}/train.csv')
label_raw = df_train['label']

train_raw = df_train.drop(['label'], axis=1) / 255.0



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(train_raw, label_raw):

    train = train_raw.loc[train_index]

    label = label_raw.loc[train_index]

    test = train_raw.loc[test_index]

    test_label = label_raw.loc[test_index]



print(len(train), len(label), len(test), len(test_label))
train = np.array(train).reshape(-1, 28,28, 1)

test = np.array(test).reshape(-1, 28,28,1)

print(train.shape)

print(test.shape)
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Dropout, Flatten, MaxPool2D, LeakyReLU

from tensorflow.keras.models import Sequential



#Data augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt



tf.__version__
datagen = ImageDataGenerator(

   # featurewise_center=True,

  #  featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=False)



datagen.fit(train)

image_iterator = datagen.flow(train)

fit,rows = plt.subplots(nrows=1, ncols=4, figsize=(18,18))

for row in rows:

    image = image_iterator.next()[0].astype('float')

    image = image.reshape(28,28)

    row.imshow(image, cmap=plt.cm.gray)

    row.axis('off')

plt.show()
model = Sequential([

    Conv2D(32, kernel_size=2, activation=tf.nn.leaky_relu, input_shape=(28,28,1), padding='same'),

    MaxPool2D(2),

    BatchNormalization(),

    LeakyReLU(),

    

    Conv2D(64, kernel_size=2, padding='same'),

    MaxPool2D(2),

    Flatten(),

    BatchNormalization(),

    LeakyReLU(),

    

    Dense(128),

    Dropout(0.2),

    BatchNormalization(),

    LeakyReLU(),

    

    Dense(256),

    Dropout(0.2),

    BatchNormalization(),

    Dense(10, activation='softmax')

])



model.summary()
optimizer=tf.keras.optimizers.Nadam(lr=0.001, decay=1e-5)

model.compile(loss=tf.losses.Huber(), optimizer=optimizer, metrics=['accuracy'])
categorical_labels = tf.keras.utils.to_categorical(label)

categorical_test_labels = tf.keras.utils.to_categorical(test_label)



earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                              patience=5, min_lr=0.00001, verbose=1)

#model.fit(train, categorical_labels, validation_split=0.1, callbacks=[earlyStop], epochs=100)
model.fit_generator(datagen.flow(train, categorical_labels, batch_size=32),

                    steps_per_epoch=len(train) // 32,

                    epochs=1000,

                    validation_data=(test, categorical_test_labels),

                    callbacks=[earlyStop, reduce_lr]

                   )
df_test = pd.read_csv(f'{BASE_DIR}/test.csv')

df_test.describe()
ids = df_test['id']

df_test = df_test.drop(['id'], axis=1)
final_test = np.array(df_test).reshape(-1, 28,28, 1) / 255
result = model.predict(final_test)
allPredictions = []

for id, prediction in zip(ids, result):

    res = np.argmax(prediction)

    allPredictions.append([id, res])

    

df_predict = pd.DataFrame(allPredictions, columns=['id', 'label'])



df_predict.to_csv('submission.csv', index=False)
print(allPredictions[:5])