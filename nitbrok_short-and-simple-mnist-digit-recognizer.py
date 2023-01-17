import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

plt.style.use('dark_background')

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.head()
test_data.head()
target = train_data['label']

train_data.drop(['label'], axis = 1, inplace = True)
print(train_data.shape)

print(test_data.shape)
target.head()
plt.figure(figsize = (15,6))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(np.array(train_data.iloc[i]).reshape(28,28), cmap = 'gray')

plt.tight_layout()
plt.figure(figsize = (15,6))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(np.array(train_data.iloc[-i]).reshape(28,28), cmap = 'gray')

plt.tight_layout()
train_data.info()
test_data.info()
train_data = np.array(train_data).reshape(len(train_data),28,28,1)

train_data = train_data/255

test_data = np.array(test_data).reshape(len(test_data),28,28,1)

test_data = test_data/255
print(train_data.shape)

print(test_data.shape)
model_1 = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), input_shape = (28,28,1), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation = 'relu'),

    tf.keras.layers.Dense(10, activation = 'softmax')

])
model_1.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_1 = model_1.fit(train_data, target, epochs=10, validation_split = 0.2)
plt.plot(history_1.history['accuracy'])

plt.plot(history_1.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best')

plt.show()
model_2 = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation = 'relu'),

    tf.keras.layers.Dense(10, activation = 'softmax')

])
model_2.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(train_data, target, epochs=10, validation_split = 0.2)
plt.plot(history_2.history['accuracy'])

plt.plot(history_2.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best')

plt.show()
model_3 = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, activation = 'relu'),

    tf.keras.layers.Dense(10, activation = 'softmax')

])
model_3.compile(optimizer ='sgd', loss ='sparse_categorical_crossentropy', metrics =['accuracy'])
history_3 = model_3.fit(train_data, target, epochs = 10, validation_split = 0.2)
plt.plot(history_3.history['accuracy'])

plt.plot(history_3.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='best')

plt.show()
model_4 = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), input_shape = (28,28,1), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation = 'relu'),

    tf.keras.layers.Dense(10, activation = 'softmax')

])
model_4.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history_4 = model_4.fit(train_data, target, epochs = 10, validation_split = 0.2)
plt.plot(history_4.history['accuracy'])

plt.plot(history_4.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='best')

plt.show()
results = model_4.predict(test_data)
results
results = np.argmax(results, axis = 1)
results
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.shape
submission.head()
submission['Label'] = results
submission.head()
submission.to_csv('my_submission.csv', index = False)