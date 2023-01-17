import tensorflow as tf

import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist #28*28 size images of hand-written digits 0-9
y_train
(x_train, y_train), (x_test,y_test) = mnist.load_data()
plt.subplot(1,2,1)

plt.imshow(x_train[-1],cmap=plt.cm.binary)

plt.subplot(1,2,2)

plt.imshow(x_train[-1])

print(x_train[0])
plt.subplot(1,2,1)

plt.imshow(x_train[-1],cmap=plt.cm.binary)

plt.subplot(1,2,2)

plt.imshow(x_train[-1])

print(x_train[0])
x_train = tf.keras.utils.normalize(x_train,axis=1)

x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))



model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)
model.save('epic_num_reader_model')

new_model = tf.keras.models.load_model('epic_num_reader_model')
predictions = new_model.predict(x_test)

print(predictions)
import numpy as np



print(np.argmax(predictions[2]))
plt.imshow(x_test[2])