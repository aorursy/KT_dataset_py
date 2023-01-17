import numpy as np

import pandas as pd



dataset = pd.read_json("/kaggle/input/planesnet/planesnet/planesnet.json")

dataset.drop("scene_ids", axis=1, inplace=True)

dataset.drop("locations", axis=1, inplace=True)

dataset.head()
pixel_columns_name = np.empty(1200, dtype=object)

channels = ["r", "g", "b"] 

for c in channels:

    for p in range(400):

        pixel_columns_name[400*channels.index(c)+p] = c+"_"+str(p)



image_data = pd.DataFrame(dataset['data'].values.tolist(), columns=pixel_columns_name)

labels = dataset.labels



image_data = pd.concat([image_data, labels], axis=1)

image_data.head()
from matplotlib import pyplot as plt

from matplotlib import colors

import random

%matplotlib inline



img_number = random.randrange(0, len(image_data.labels.tolist()))

print("Image number:", img_number, "with label:", image_data.labels[img_number])



plt.subplot(1, 3, 1)

plt.imshow(np.array(image_data.iloc[img_number,0:400].values.tolist()).reshape(20,20), cmap='Reds_r')

plt.title("R")

plt.subplot(1, 3, 2)

plt.imshow(np.array(image_data.iloc[img_number,401:801].values.tolist()).reshape(20,20), cmap='Greens_r')

plt.title("G")

plt.subplot(1, 3, 3)

plt.imshow(np.array(image_data.iloc[img_number,801:1201].values.tolist()).reshape(20,20), cmap='Blues_r')

plt.title("B")

plt.show()
y = image_data.labels

image_data.drop("labels", inplace=True, axis=1)

X = image_data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train.to_numpy().reshape(-1, 20,20, 3) # 20*20, 3 channels (R-G-B)

X_test = X_test.to_numpy().reshape(-1, 20,20, 3) # 20*20, 3 channels (R-G-B)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train = X_train / 255.

X_test = X_test / 255.
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=60,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
X_train,X_valid,y_train,y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=13)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import keras

import tensorflow as tf

from keras.models import Sequential,Input,Model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU



from tensorflow.python.client import device_lib 

device_lib.list_local_devices() # let's list all available computing devices
batch_size = 64

epochs = 20

num_classes = 2 # in fact, each image could or could not contain a plane
model = Sequential()



model.add(Conv2D(64, kernel_size=(3, 3),activation='linear',input_shape=(20,20,3),padding='same'))

model.add(BatchNormalization())

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D((4, 4),padding='same'))

model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

model.add(BatchNormalization())

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(4, 4),padding='same'))

model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))

model.add(BatchNormalization())

model.add(LeakyReLU(alpha=0.1))                  

model.add(MaxPooling2D(pool_size=(4, 4),padding='same'))

model.add(Flatten())

model.add(Dense(128, activation='linear'))

model.add(LeakyReLU(alpha=0.1))                  

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()
with tf.device('/GPU:0'):

    model_train = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, y_valid))
plt.figure(figsize=(7, 7), dpi=80)

plt.subplot(2,1,1)

plt.title("Training History - Accuracy")

plt.plot(range(epochs), model_train.history["accuracy"], label="accuracy", color="red")

plt.scatter(range(epochs), model_train.history["val_accuracy"], label="val_accuracy")

plt.xticks(range(0,epochs,1))

min_y = min(np.min(model_train.history["val_accuracy"]), np.min(model_train.history["accuracy"]))

plt.yticks(np.linspace(min_y-0.1,1,11))

plt.legend()





plt.subplot(2,1,2)

plt.title("Training History - Loss")

plt.plot(range(epochs), model_train.history["val_loss"], label="val_loss", color="red")

plt.scatter(range(epochs), model_train.history["loss"], label="loss")

plt.xticks(range(0,epochs,1))

max_y = max(np.max(model_train.history["val_loss"]), np.max(model_train.history["loss"]))

plt.yticks(np.linspace(0,max_y+0.1,11))

plt.legend()
test_eval = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', test_eval[0])

print('Test accuracy:', test_eval[1]*100, "%")
from sklearn.metrics import classification_report



y_pred = np.argmax(np.round(model.predict(X_test)), axis=1)



target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_test, y_pred, target_names=target_names))
from sklearn.metrics import confusion_matrix

import seaborn as sn



cf = confusion_matrix(y_test, y_pred)

sn.heatmap(cf, annot=True)

plt.title("Confusion Matrix")