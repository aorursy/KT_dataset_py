# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import Input
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base_dir = '../input/digit-recognizer/'
train_df = pd.read_csv(base_dir + 'train.csv')
test_df = pd.read_csv(base_dir + 'train.csv')

train_y = train_df.pop("label")
test_labels = test_df.pop("label")

train_y = np_utils.to_categorical(train_y, num_classes = 10)
print(train_df.shape)
train_x = train_df.values.reshape(-1,28,28,1)
test_data = test_df.values.reshape(-1,28,28,1)
train_data, val_data, train_labels, val_labels = train_test_split(train_x, train_y, test_size = 0.1)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(train_data)
# Normalize the data
val_data = val_data / 255.0
test_data = test_data / 255.0


input_tensor = Input(shape=(28,28,1,))
x = layers.Conv2D(32,(3,3,), activation='relu')(input_tensor)
x = layers.Conv2D(32,(3,3,), activation='relu')(x)
x = layers.MaxPool2D(2,2)(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(64,(3,3,), activation='relu')(x)
x = layers.MaxPool2D(2,2)(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation = 'relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = models.Model(input_tensor, output_tensor)
model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
print(len(train_data),len(train_labels))
print(len(val_data),len(val_labels))
callbacks_list = [
keras.callbacks.EarlyStopping(
monitor='acc',
patience=1,
),
keras.callbacks.ModelCheckpoint(
filepath='my_model.h5',
monitor='val_loss',
save_best_only=True,),
    
keras.callbacks.ReduceLROnPlateau(
monitor='val_loss',
factor=0.1,
patience=5,
)
]
history = model.fit_generator(train_datagen.flow(train_data,train_labels, batch_size=32),
                              epochs = 35, validation_data = (val_data,val_labels),callbacks= callbacks_list)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
my_model = keras.models.load_model('./my_model.h5')
results = my_model.predict(test_data)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
submission = pd.read_csv("./submission.csv")
submission.head(20)

