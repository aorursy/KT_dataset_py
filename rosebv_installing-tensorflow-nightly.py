!pip show tensorflow
!pip install tf-nightly
import numpy as np

import tensorflow as tf



x_train = np.random.random((100, 28, 28))

y_train = np.random.randint(10, size=(100, 1))

x_test = np.random.random((20, 28, 28))

y_test = np.random.randint(10, size=(20, 1))



model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy'])



model.fit(x_train, y_train, epochs=1)

model.evaluate(x_test, y_test)