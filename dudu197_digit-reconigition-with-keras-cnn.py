import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.sample()
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols)
X = np.array(data_train.iloc[:, 1:])
y = np.array(data_train.iloc[:, 0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)
X_train[0].shape
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
X_train = X_train / 255
X_test = X_test / 255
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=28)
test_loss, text_acc = model.evaluate(X_test, y_test)
print('Test loss: ', test_loss)
print('Test accuracy: ', text_acc)
history.history
accuracy = history.history['acc']
#val_accuracy = history.history['val_acc']
loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#plt.bar(epochs, accuracy, label='Training accuracy')
#plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
predictions = model.predict(X_test)
j = np.random.randint(0, len(predictions))
j
plt.figure(figsize=(10,10))
for i in range(25):
    j = np.random.randint(0, len(predictions))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[j], cmap=plt.cm.binary)
    plt.xlabel("{} - {:2.0f}%".format(predictions[j].argmax(), (predictions[j].max() * 100)))
data_test = data_test / 255
data_test = data_test.values.reshape(-1, img_rows, img_cols)
plt.figure()
plt.imshow(data_test[0])
plt.colorbar()
plt.grid(False)
results = model.predict(data_test)
plt.figure(figsize=(10,10))
for i in range(25):
    j = np.random.randint(0, len(results))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_test[j], cmap=plt.cm.binary)
    plt.xlabel("{} - {:2.0f}%".format(results[j].argmax(), (results[j].max() * 100)))
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
results.sample()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)