import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
from os import getcwd
import csv
train=pd.read_csv('../input/digit-recognizer/train.csv')
test=pd.read_csv('../input/digit-recognizer/test.csv')
train.head()


test.head()
train_x=train.drop(['label'],axis=1)
train_x.head()
train_x=train_x.values
train_x
train_x=np.reshape(train_x,(42000,28,28))
train_x.shape
train_y=train[train.columns[0:1]]
train_y.head()
train_y=train_y.values
train_y
train_y.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.5, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)
print(x_test.shape)
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1.0/255)
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    
    tf.keras.layers.Flatten(),
    
    # 128 neuron hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
model.summary()
model.compile(optimizer = tf.optimizers.Adam(learning_rate=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=32),
                              #steps_per_epoch=len(x_train) / 32,
                              epochs=30,
                              validation_data=validation_datagen.flow(x_test, y_test, batch_size=32),
                              #validation_steps=len(x_test) / 32
                             )

model.evaluate(x_test, y_test)
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
test.head()
test_x=test.values
test_x
test_x=np.reshape(test_x,(28000,28,28))
print(test_x.shape)
test_x=np.expand_dims(test_x,axis=3)
test_x.shape
result=model.predict(test_x)
result[0]
result1=np.argmax(result,axis=1)
result1
resultfinal=pd.Series(result1,name='Label')
resultfinal
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),resultfinal],axis = 1)

submission.to_csv("mnist_submission.csv",index=False)
