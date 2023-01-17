# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np



import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_datasets as tfds



import matplotlib.pyplot as plt



print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("Hub version: ", hub.__version__)

print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
df_true=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

df_Fake=pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
df_true['state'] = 1

df_Fake['state'] =0

data = pd.concat([df_Fake,df_true],axis=0)

data.head()

y = data['state'].values

X = data.drop(['state', 'date'], axis = 1)

X.head()
y = data['state'].values

X = data.drop(['state', 'date'], axis = 1)

X.head()
X.shape
model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[], 

                           dtype=tf.string, trainable=True)

hub_layer(X.text[:3])
model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1))



model.summary()

model.compile(optimizer='adam',

              loss=tf.losses.BinaryCrossentropy(from_logits=True),

              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

#X['combined']=X.apply(lambda x:'%s_%s' % (X['text'],X['subject']),axis=1)

#X.combine
x_val = X.text[:10000]

partial_x_train = X.text[10000:]



y_val = y[:10000]

partial_y_train = y[10000:]
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=40,

                    batch_size=512,

                    validation_data=(x_val, y_val),

                    verbose=1)

results = model.evaluate(x_val, y_val)



print(results)
history_dict = history.history

history_dict.keys()
acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # clear figure



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
