import numpy as np 

import pandas as pd

import tensorflow.keras as tf
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df = pd.DataFrame(train_data)#.head(32000)

train_images = df.drop(["label"], axis=1).values

train_labels = df["label"].values



train_images = train_images.reshape(42000, 28, 28, 1)

#print(train_labels)
model = tf.models.Sequential()



model.add(tf.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(tf.layers.MaxPooling2D((2, 2)))

model.add(tf.layers.Conv2D(128, (3, 3), activation='relu'))

model.add(tf.layers.MaxPooling2D((2, 2)))



model.add(tf.layers.Flatten())

model.add(tf.layers.Dense(64, activation='relu'))

model.add(tf.layers.Dense(10, activation='softmax'))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



#model.summary()

res = model.fit(train_images, train_labels, epochs=4, validation_split=0.35)
df = pd.DataFrame(test_data)

test_images = df.values



test_images = test_images.reshape(28000, 28, 28, 1)
res = np.argmax(model.predict(test_images), axis=1)
np.savetxt("./res.csv", res, delimiter=",")