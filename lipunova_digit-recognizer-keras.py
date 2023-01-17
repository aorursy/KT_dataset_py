import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.utils import to_categorical

from tensorflow.python.keras import layers, models

from tensorflow.python.keras.regularizers import l2

from tensorflow.python.keras.losses import categorical_crossentropy



%matplotlib inline
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')



y = to_categorical(train.label, 10)

x = train.drop('label', axis=1).values.reshape(train.shape[0], 28, 28, 1) / 255

x_test = test.values.reshape(test.shape[0], 28, 28, 1) / 255
model = models.Sequential()

model.add(layers.Conv2D(16, kernel_size=(9, 9), kernel_initializer='he_normal',

                        kernel_regularizer=l2(0.005), input_shape=(28, 28, 1)))

model.add(layers.Conv2D(32, kernel_size=(7, 7), strides=2, kernel_initializer='he_normal',

                        kernel_regularizer=l2(0.005), activation='relu'))

model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, kernel_size=(4, 4), kernel_initializer='he_normal',

                        kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Conv2D(128, kernel_size=(4, 4), kernel_initializer='he_normal',

                        kernel_regularizer=l2(0.0001), activation='relu'))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(512, kernel_initializer='he_normal', activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

hist = model.fit(x, y, batch_size=64, epochs=11, steps_per_epoch=len(x)*0.7/64, validation_split=0.3)
print('Train set: ', ', '.join([str(acc) for acc in hist.history['accuracy']]))

print('Validation set: ', ', '.join([str(acc) for acc in hist.history['val_accuracy']]))



fig, ax = plt.subplots(1, 2, figsize=(20, 5))

ax[0].plot(hist.history['accuracy'])

ax[0].set_title('Train set', fontsize=12)

ax[0].set_ylabel('accuracy')

ax[0].set_xlabel('epoch')

ax[1].plot(hist.history['val_accuracy'])

ax[1].set_title('Validation set', fontsize=12)

ax[1].set_ylabel('accuracy')

ax[1].set_xlabel('epoch')
fig, ax = plt.subplots(1, 10, constrained_layout=True, figsize=(20, 20))

pred = np.argmax(model.predict(x_test), axis=1)

for i in range(10):

    ax[i].imshow(x_test[i].reshape(28, 28))

    ax[i].set_xlabel('predicted:' + str(pred[i]), fontsize=18)
result = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

result['Label'] = pred

result.to_csv('submission.csv', index=False)