import tensorflow as tf

import matplotlib.pyplot as plt

import cv2



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
mnist_dataset = tf.keras.datasets.mnist

(train_image_mnist, train_label_mnist), (test_image_mnist, test_label_mnist) = mnist_dataset.load_data()
train_image_mnist = tf.keras.utils.normalize(train_image_mnist, axis=1)

test_image_mnist = tf.keras.utils.normalize(test_image_mnist, axis=1)
plt.imshow(train_image_mnist[19], cmap='gray')
model = Sequential()



model.add(CuDNNLSTM(128, input_shape= train_image_mnist.shape[1:], 

          return_sequences = True))

model.add(Dropout(0.2))



model.add(CuDNNLSTM(128))

model.add(Dropout(0.2))



model.add(Dense(32, activation = 'relu'))

model.add(Dropout(0.2))



model.add(Dense(10, activation = 'softmax'))
model.compile(loss='sparse_categorical_crossentropy', 

              optimizer=tf.keras.optimizers.Adam(lr= 1e-3, decay=1e-5),

             metrics=['accuracy'])
model.fit(train_image_mnist, train_label_mnist, 

          epochs=10, validation_data=(test_image_mnist, test_label_mnist))