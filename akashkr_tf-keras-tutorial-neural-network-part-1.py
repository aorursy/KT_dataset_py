import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

%matplotlib inline
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(f'Shape of the training data: {x_train.shape}')
print(f'Shape of the training target: {y_train.shape}')
print(f'Shape of the test data: {x_test.shape}')
print(f'Shape of the test target: {y_test.shape}')
print(y_train)
# Let's plot the first image in the training data and look at it's corresponding target (y) variable.
plt.imshow(x_train[0], cmap='gray')
print(f'Target variable is {y_train[0]}')
# Setting custom printwidth to print the array properly
np.set_printoptions(linewidth=200)
print(x_train[0])
# Normalizing the data
# each element of nested list/array in python is divided by using a simple division operator on the list/array
x_train = x_train/255
x_test = x_test/255
print(x_train[0])
# Creating the architecture of model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=3)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f'Validation loss: {val_loss}')
print(f'Validation accuracy: {val_acc}')
# Callback class which checks on the logs when the epoch ends
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.05):
      print("\nReached Minimal loss so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Pass callbacks parameter while training
model.fit(x_train, y_train, epochs=50, callbacks=[callbacks])