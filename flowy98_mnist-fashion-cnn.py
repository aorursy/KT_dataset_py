# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/fashion-mnist_train.csv')

df_train.head()
labels = df_train.iloc[:,0]

labels.head()
features = df_train.iloc[:,1:]

features.head()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



def get_name_of_label(label):

    return class_names[label]
plt.figure(figsize=(12, 8))

for i in range(1, 6):

    plt.subplot(150+i)

    plt.imshow(np.reshape(features.loc[i].values/255, (28, 28)), 'gray')

    plt.title(get_name_of_label(labels[i]))

    plt.xticks([])

    plt.yticks([])
features = features.values.reshape(-1, 28, 28, 1) 
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
history = model.fit(features/255, labels, epochs=10, batch_size=32, validation_split=0.2, shuffle=True)
plt.figure(figsize=(18, 6))



plt.subplot(231)

plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend()



plt.subplot(232)

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend()
df_test = pd.read_csv('../input/fashion-mnist_test.csv')

df_test.head()
test_labels = df_test['label']

test_features = df_test.iloc[:, 1:]
test_loss, test_accuracy = model.evaluate(test_features.values.reshape(-1, 28, 28, 1), test_labels)
predictions = []

for i in range(0, 20):

    predictions.append(model.predict(test_features.loc[i].values.reshape(-1, 28, 28, 1)))
plt.figure(figsize=(12, 8))

for i in range(0, 20):

    plt.subplot(4, 5, i+1)

    plt.imshow(np.reshape(test_features.loc[i].values/255, (28, 28)), 'gray')

    plt.xticks([])

    plt.yticks([])

    prediction = np.argmax(predictions[i])

    plt.title('{} | {}'.format(get_name_of_label(test_labels[i]), get_name_of_label(prediction)))