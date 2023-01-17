!pip install tensorflow==2.0-rc1
import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_data_raw = pd.read_csv("../input/digit-recognizer/train.csv")

test_data_raw = pd.read_csv("../input/digit-recognizer/test.csv")

train_data_raw.describe(), test_data_raw.describe()
train_labels = train_data_raw.pop("label")
train_data_raw.shape, test_data_raw.shape
train_data_norm = train_data_raw.to_numpy()/255.0

test_data_norm = test_data_raw.to_numpy()/255.0
x_train = train_data_norm.reshape(len(train_data_norm), 28, 28, 1)

y_train = train_labels.to_numpy()

x_test = test_data_norm.reshape(len(test_data_norm), 28, 28, 1)
x_train.shape, x_test.shape
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3),  activation='relu', input_shape=(28, 28, 1)))

model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3),  activation='relu'))

model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3),  activation='relu'))
model.summary()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, 'relu'))

model.add(tf.keras.layers.Dense(10, 'softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=32)
plt.plot(history.history['accuracy'], label='accuracy')

# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')
pred = model.predict(x_test)
y_test = np.argmax(pred, axis=1)
result_df = pd.DataFrame(y_test, columns=['label'], index=range(1,len(y_test)+1)).rename_axis('ImageId')
result_df.head()
result_df.to_csv("submission.csv")