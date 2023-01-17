import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df.shape
train_labels = df["label"]

train_labels.head()
train_data = df.loc[:, df.columns != 'label']

train_data.head()
train_data = train_data.to_numpy()

train_labels = train_labels.to_numpy()
train_data = train_data / 255
train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))

train_data.shape
filters = 64

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=filters, kernel_size=(5,5), strides=2, padding='same', activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),



    tf.keras.layers.Conv2D(filters=2*filters, kernel_size=(3,3), activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=2*filters, kernel_size=(3,3), activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=2*filters, kernel_size=(5,5), strides=2, padding='same', activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),



    tf.keras.layers.Conv2D(filters=4*filters, kernel_size=(4,4), activation=tf.nn.relu),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.4),

    

    tf.keras.layers.Dense(64, activation=tf.nn.relu),

    tf.keras.layers.Dense(32, activation=tf.nn.relu),

    tf.keras.layers.Dense(16, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,

                                                               height_shift_range=0.1,

                                                               rotation_range=5,

                                                               zoom_range=0.1)
history = model.fit(dataGenerator.flow(train_data, train_labels, batch_size=32), epochs=100, validation_data=(train_data, train_labels))
plt.figure(figsize=[8,6])

plt.plot(history.history['accuracy'],'r',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)

plt.show()
df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df_test.shape
test_data = df_test.to_numpy()

test_data = test_data / 255

test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

test_data.shape
predictions = model.predict(test_data)

predictions = np.asarray([np.argmax(prediction) for prediction in predictions])

predictions.shape
df_predictions = pd.DataFrame(predictions).rename(columns={0: "Label"})

df_predictions.index.names = ['ImageId']

df_predictions.index += 1

df_predictions.head()
df_predictions.shape

df_predictions.to_csv("predictions.csv")