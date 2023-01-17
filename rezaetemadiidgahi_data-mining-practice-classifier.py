import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub



print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("Hub version: ", hub.__version__)

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.head()
print("befor convert: ", type(train_df["target"].values[1]))

train_df["target"] = train_df["target"].astype(float)

print("after convert: ", type(train_df["target"].values[1]))
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=True)
hub_layer(train_df["text"].values[:3])
model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),

              metrics=['accuracy'])
history = model.fit(train_df["text"].values,

                    train_df["target"].values,

                    epochs=10,

                    verbose=1)
import matplotlib.pyplot as plt



history_dict = history.history

acc = history_dict['accuracy']

loss = history_dict['loss']



epochs = range(1, len(acc) + 1)



# b is for "solid blue line"

plt.plot(epochs, loss, 'b', label='Training loss')

plt.title('Training loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # clear figure



plt.plot(epochs, acc, 'b', label='Training acc')

plt.title('Training accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')



plt.show()
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = np.round(model.predict(test_df["text"].values)).astype(int)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)