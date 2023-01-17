import tensorflow as tf





# Import MNIST dataset.

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Normalize the data.

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)



# Create the model.

model = tf.keras.models.Sequential()



# Add a flattening layer.

model.add(tf.keras.layers.Flatten())



# Dense layers.

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))



# Output layer.

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



# Compile the model.

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)



# Fit it.

model.fit(x_train, y_train, epochs=3)
import numpy as np

import matplotlib.pyplot as plt

import random



# Randomly select an item and label.

i = random.randint(0, len(x_test) - 1)

item = x_test[i]

label = y_test[i]

# Plot an image for that item.

plt.imshow(item, cmap=plt.cm.binary)

# Print out the prediction and true value.

prediction = model.predict([[item]])

print('Prediction: ', np.argmax(prediction))

print('True value: ', label)
val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_loss, val_acc)