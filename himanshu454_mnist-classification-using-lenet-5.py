from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
image_index = 7757  
print(y_train[image_index])  
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()
y_train.shape , y_test.shape , x_train.shape , x_test.shape

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train.shape , y_test.shape , x_train.shape , x_test.shape
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalize
x_train /= 255
x_test /= 255
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5,5), input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(1024, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x=x_train,y=y_train, epochs=10 , batch_size = 128)
test_error_rate = model.evaluate(x_test, y_test, verbose=0)
print(" mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
model.save("trained_model.h5")
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")