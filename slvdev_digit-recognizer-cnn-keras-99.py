%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from random import randrange
df_train = pd.read_csv('../input/train.csv')

print(f'The train set contain {df_train.shape[0]} examples')

df_train.head(3)
X_train = df_train.drop('label', axis = 1)

y_train = df_train['label']
digits = y_train.unique()

values = y_train.value_counts()



plt.bar(digits, values)

plt.title('Train set')

plt.xlabel('Digit')

plt.ylabel('Examples count')

plt.xticks(np.arange(len(digits)))

plt.show()
X_train = X_train / 255
X_train = X_train.values.reshape(-1,28,28,1)
rnd_digit = randrange(X_train.shape[0])

img = X_train[rnd_digit][:,:,0]

label = y_train[rnd_digit]

plt.title(f'This is number {label}')

plt.axis('off')

plt.imshow(img, cmap=plt.cm.binary)
y_train = to_categorical(y_train)

y_train.shape
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)
train_datagen = ImageDataGenerator(

        rotation_range=10,

        zoom_range = 0.1, 

        width_shift_range=0.1,

        height_shift_range=0.1,

        shear_range=0.1,

        )



train_datagen.fit(X_train)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.2),

    

    tf.keras.layers.Flatten(),

    

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    

    tf.keras.layers.Dense(10, activation='softmax')

])





model.compile(loss='categorical_crossentropy',

              optimizer=Adam(0.0001),

              metrics=['acc'])
model.summary()
batch_size = 32

history = model.fit_generator(

        train_datagen.flow(X_train, y_train, batch_size=batch_size),

        epochs=20,

        validation_data=(X_val, y_val),

        )
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

y_pred_cl = np.argmax(y_pred, axis = 1)

y_true = np.argmax(y_test, axis = 1)



confusion_matrix(y_true, y_pred_cl)
test = pd.read_csv('../input/test.csv')

test = test / 255

test = test.values.reshape(-1,28,28,1)

print(f'The test set contain {test.shape[0]} examples')
pred = model.predict(test)

pred = np.argmax(pred, axis = 1)
pred_csv = pd.DataFrame(pred, columns= ['Label'])

pred_csv.index += 1

pred_csv.head()
pred_csv.to_csv('submission.csv', index_label='ImageId' )