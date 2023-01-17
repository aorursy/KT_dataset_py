import tensorflow as tf



from tensorflow import keras

from tensorflow.keras.layers import Dense

from tensorflow.keras import Sequential



import numpy as np
display(f"TensorFlow: v{tf.VERSION}; Keras: v{keras.__version__}")
model = Sequential()

model.add(Dense(64, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(10, activation="softmax"))



model.compile(optimizer=tf.train.AdamOptimizer(0.001),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
data = np.random.random((1000, 32))

labels = np.random.random((1000, 10))



val_data = np.random.random((100, 32))

val_labels = np.random.random((100, 10))



model.fit(data, labels, epochs=10, batch_size=32,

          validation_data=(val_data, val_labels))
data = np.random.random((1000, 32))

labels = np.random.random((1000, 10))



model.evaluate(data, labels, batch_size=32)