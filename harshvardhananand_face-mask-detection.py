# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras import layers
data = np.load('/kaggle/input/face-mask-detection-dataset/CleanedData/data.npy')
target = np.load('/kaggle/input/face-mask-detection-dataset/CleanedData/target.npy')

target = tf.keras.utils.to_categorical(target) # converting [1,0,1,1,0...] to [[1,0],[0,1],[1,0]] i.e categorical

target
plt.imshow(data[890],cmap='gray')
target[890]
# we need to create a 3D image since our image have only one channel as its a greyscale img.

# so we will just replicate the image 3 times to create a 3d image.

# for this process tf.image.grayscale_to_rgb comes handy.

# https://www.tensorflow.org/api_docs/python/tf/image/grayscale_to_rgb

# why RESHAPE - https://github.com/tensorflow/tensorflow/issues/26324 

# preprocessing

odata = data.copy()

ndata = []

for i in odata:

  image = i.reshape((*i.shape,1)) # as tf.image.grayscale_to_rgb requires last dimension to be 1, see why reshape link

  image = tf.convert_to_tensor(image)  # as tf.image.grayscale_to_rgb requires tensor for processing.

  ndata.append(tf.image.grayscale_to_rgb(image).numpy()/255.) # .numpy will convert dtype to numpy from tf



data = ndata.copy()
from sklearn.model_selection import train_test_split
np.shape(data)
trainx, testx, trainy, testy = train_test_split(data,

                                                target,

                                                test_size=0.15,

                                                random_state=345,

                                                shuffle=True)

# we need to convert list to np array for tensorflow

trainx = np.array(trainx)

testx = np.array(testx)
trainy.shape
plt.figure(figsize=[30,30])

for i in np.arange(1,10):

    plt.subplot(int(f"19{i}"))

    plt.imshow(trainx[np.random.randint(0,1403)], cmap='gray')
plt.subplot(221)

plt.imshow(trainx[np.random.randint(0,1440)], cmap='gray')

plt.subplot(222)

plt.imshow(trainx[np.random.randint(0,1440)], cmap='gray')

plt.subplot(223)

plt.imshow(trainx[np.random.randint(0,1440)], cmap='gray')

plt.subplot(224)

plt.imshow(trainx[np.random.randint(0,1440)], cmap='gray')
trainy[0]
img_shape = trainx[0].shape

img_shape  
model=Sequential()



model.add(layers.Conv2D(32,(3,3),input_shape=img_shape))

# model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))



model.add(layers.Conv2D(64,(3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))



model.add(layers.Conv2D(128,(3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))



model.add(layers.Conv2D(256,(3,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))





model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dropout(0.4))



model.add(layers.Dense(2,activation='softmax'))

#The Final layer with two outputs for two categories





adam = tf.keras.optimizers.Adam(0.001)

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
model.summary()
# monitor = on which basis we have to select best model

# verbose = 0 i.e no message, 1 i.e update message.

model_saving_path = r'/tmp/checkpoint/cnn.model'

save_model = tf.keras.callbacks.ModelCheckpoint(model_saving_path,

                                                monitor='val_acc',

                                                save_best_only=True,

                                                mode='max',

                                                verbose=1)
history = model.fit(x=trainx,

                    y=trainy,

                    batch_size=100,

                    epochs=50,

                    callbacks=[save_model],

                    validation_split=0.2,

                    verbose=2,

                    shuffle=True)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loss','val_loss'])
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['accuracy', 'val_accuracy'])
model.evaluate(testx, testy)
model.save('cnn.h5')