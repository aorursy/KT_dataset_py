import numpy as np 
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

X = X.reshape(X.shape[0], img_rows, img_cols, 1)
X = X.astype('float32')
X /= 255

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_test = X_test.astype('float32')
X_test /= 255
plt.figure(figsize=(8, 8))
for digit_num in range(0,25):
    plt.subplot(5,5,digit_num +1)
    plt.imshow(X[digit_num].reshape(28,28), cmap = "gray")
    plt.axis('off')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 256
num_classes = 10
epochs = 50

#input image dimensions
img_rows, img_cols = 28, 28

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
import tensorflow as tf
from keras.callbacks import EarlyStopping

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 5, mode = max)
history = model.fit(X, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split= 0.2, callbacks = [callback])

score = model.evaluate(X_test, y_test, verbose=1)
print('Loss:', score[0])
print('Accuracy:', score[1])
import matplotlib.pyplot as plt
%matplotlib inline
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Predictions of the classes
predicted_classes = model.predict_classes(X_test)

#True Classes
y_true = data_test.iloc[:, 0]


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names, digits = 4))
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_true, predicted_classes)
print(cf)
rowsum = cf.sum(axis=1, keepdims=True)
cf_percent = cf*100 / rowsum
np.fill_diagonal(cf_percent, 0)
plt.figure(figsize=(7, 7))
plt.matshow(cf_percent, cmap='hot_r', fignum = 1)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#0 - T-shirt, 2 - pullover, 4- coat, 6 - Shirt