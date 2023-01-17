import tensorflow as tf
import numpy as np
def load_data():
    with np.load("../input/mnist.npz") as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train),(x_test, y_test) = load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])
model.fit(x=x_train,y=y_train,batch_size=20,epochs=5,validation_data=(x_test,y_test))

model.evaluate(x=x_test,y=y_test)
