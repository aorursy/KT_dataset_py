import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
data_train.columns
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
x_train = np.array(data_train.drop(["label"],axis=1))
y_train = np.array(data_train["label"])
x_test = np.array(data_test.drop(["label"],axis=1))
y_test = np.array(data_test["label"])
x_train.shape
x_test.shape
data_train["label"].value_counts()

x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(x_train, 
                    y_train,
                   epochs = 50,validation_data=(x_test,y_test))
model.evaluate(x_test, y_test, verbose=0)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss =  history.history['loss']
val_acc=history.historytory["val_accuracy"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='validation Loss')
plt.title('Training and validation  loss')
plt.legend()

plt.show()
model1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model1.compile(optimizer = 'rmsprop', 
              loss = 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history = model1.fit(x_train, 
                    y_train,
                   epochs = 50,validation_data=(x_test,y_test))
model1.evaluate(x_test, y_test, verbose=1)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss =  history.history['loss']
val_acc=history.history["val_accuracy"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='validation Loss')
plt.title('Training and validation  loss')
plt.legend()

plt.show()

model3 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  #tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  #tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model3.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history = model3.fit(x_train, 
                    y_train,
                   epochs = 25,validation_data=(x_test,y_test))
model3.evaluate(x_test, y_test, verbose=1)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss =  history.history['loss']
val_acc=history.history["val_accuracy"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='validation Loss')
plt.title('Training and validation  loss')
plt.legend()

plt.show()
from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras import models
model = models.Sequential()
model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))
model.add(LeakyReLU())
model.add(Conv2D(32,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation="sigmoid"))


model.compile(optimizer = 'rmsprop', 
              loss = 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(x_train, 
                    y_train, batch_size=256,
                   epochs = 25,validation_data=(x_test,y_test))
model.evaluate(x_test, y_test, verbose=1)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss =  history.history['loss']
val_acc=history.history["val_accuracy"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='validation Loss')
plt.title('Training and validation  loss')
plt.legend()

plt.show()
test_im = x_train[1578]
plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')
plt.show()