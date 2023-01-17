import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.utils import to_categorical
#os.chdir("../input/sign-language-mnist/")
train = np.genfromtxt("sign_mnist_train.csv", delimiter = ",")
test = np.genfromtxt("sign_mnist_test.csv", delimiter = ",")
from keras.utils import to_categorical
Y_train = train[1:,0].reshape(27455,1)
Y_train = to_categorical(Y_train)
Y_test = test[1:,0].reshape(7172,1)
Y_test = to_categorical(Y_test)
X_train = train[1:,1:].reshape(27455,28,28,1)
X_test = test[1:,1:].reshape(7172,28,28,1)

print("Number of training examples = " + str(X_train.shape[0]))
print("Number of test examples = " + str(X_test.shape[0]))
print("Shape of X_train = " + str(X_train.shape))
print("Shape of X_test = " + str(X_test.shape))
print("Shape of Y_train = " + str(Y_train.shape))
print("Shape of Y_test = " + str(Y_test.shape))
# Check any example from the training set by changing value of m

Image = train[1:,1:].reshape(27455,28,28)
m = 555
plt.subplot(1,1,1)
plt.grid(False)
print(Y_train[m])
plt.imshow(Image[m], cmap=plt.cm.binary)
X_train, X_test = X_train / 255, X_test / 255
import tensorflow as tf
from tensorflow.keras import layers
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(25, activation="softmax"))
model.summary()
model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=20, 
                    validation_data=(X_test, Y_test))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()
predictions = model.predict_classes(X_test)
my_submission = pd.DataFrame(predictions.astype(float))
os.chdir("/kaggle/working")
my_submission.to_csv('submission.csv', index=False)
