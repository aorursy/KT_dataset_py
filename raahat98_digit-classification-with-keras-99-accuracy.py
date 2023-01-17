import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
%matplotlib inline
from sklearn.model_selection import  train_test_split
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1)
del train
X_train.shape, y_train.shape
test.shape
X_train = X_train/255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test/255.0
test = test.values.reshape(-1, 28, 28, 1)
y_train[9]
g = sns.countplot(y_train)
y_train = to_categorical(y_train, num_classes = 10)
y_train[9]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 20, batch_size = 128, validation_data = (X_val, y_val), verbose = 2)
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, 21)

plt.plot(epochs, loss, 'ko', label = 'Training Loss')
plt.plot(epochs, val_loss, 'k', label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.plot(epochs, acc, 'yo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'y', label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
results = model.predict(test)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = 'Label')
results
submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)
submission.to_csv("MNIST_Dataset_Submissions.csv", index = False)