import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.utils.np_utils import to_categorical
def get_data(data_set, label_column='label'):
    # data_set:   pd DataFrame
    data_X = data_set.drop(columns=label_column)
    data_y = data_set[label_column]
    return data_X, data_y
# read data
train_set = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test_set = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
dig_set = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
X_train, y_train = get_data(train_set, label_column='label')
X_dig, y_dig = get_data(dig_set, label_column='label')
X_train.shape, y_train.shape, X_dig.shape, y_dig.shape

X_train = X_train.to_numpy().reshape(60000, 28, 28, 1)
y_train = y_train.to_numpy().reshape(60000, 1)
X_dig = X_dig.to_numpy().reshape(10240, 28, 28, 1)
y_dig = y_dig.to_numpy().reshape(10240, 1)
test_set = test_set.drop(columns='id').to_numpy().reshape(5000, 28, 28, 1)
X_train = X_train / 255.0
X_dig = X_dig / 255.0
test_set = test_set / 255.0
y_train_cat = to_categorical(y_train)
y_dig_cat = to_categorical(y_dig)
class_names = np.arange(10)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i,:,:,0], cmap='gray')
    # y_train is in one-hot format
    plt.xlabel(np.where(y_train_cat[i]==1)[0][0])
plt.show()
# conv layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss=['categorical_crossentropy'],
              metrics=['accuracy'])
history = model.compile(optimizer='adam',
                        loss=['categorical_crossentropy'],
                        metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, epochs=10, 
                    validation_data=(X_dig, y_dig_cat))
history.history
model.save('ccyy')
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
y_pred = model.predict_classes(test_set)
y_pred_pd = pd.DataFrame({'id': np.arange(len(y_pred)),
                          'label': y_pred})
y_pred_pd.to_csv('./y_pred.csv', sep=',', index=False)