from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np


with np.load("../input/keras-mnist-amazonaws-npz-datasets/mnist.npz") as f:
        X_train, y_train = f['x_train'], f['y_train']
        X_test, y_test = f['x_test'], f['y_test']

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#Making each y's shape (1,10)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Shape of one output column",y_train[0])
print("X_train shape:",X_train.shape,"  y_train shape:", y_train.shape)
print("X_test shape:",X_test.shape,"  y_test shape:", y_test.shape)
plt.imshow(X_train[1])
#create model
model = Sequential()

#add model layers
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))

#compile model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#train model
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=5)