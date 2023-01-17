import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
#Loading Dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#Printing dataset size

print('Dataset size : ')
print('Train ', x_train.shape)
print('Test ', x_test.shape)
#Reshaping images to 28px by 28px and 1 channel

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print('Reshaped image ', x_train[0].shape)
#One-Hot-Encoding

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print('One hot encoded label ', y_train_one_hot[0])
#Building CNN 
model = Sequential()
#First Layer
model.add(Conv2D(64, 3, activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

#Second Layer
model.add(Conv2D(32, 3, activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

#Flatten and Output Dense Layer
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
#Compiling Model
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
#Training Model
hist = model.fit(x_train, y_train_one_hot,
                 validation_split = 0.2,
                 epochs = 4)
#Visualizing Model Accuracy
plt.title('Model Accuracy')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.show()
score = model.evaluate(x_test, y_test_one_hot, batch_size=128)
print(y_test[0:5])
predictions = model.predict(x_test[0:5])
predictions = np.argmax(predictions, axis=1)

print(predictions)