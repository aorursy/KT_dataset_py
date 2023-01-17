from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D

from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.optimizers import Adam, SGD, RMSprop



import tensorflow as tf



import os

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline
DATASET_DIR = "../input/covid-19-x-ray-10000-images/dataset"



os.listdir(DATASET_DIR)

import glob

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



normal_images = []

for img_path in glob.glob(DATASET_DIR + '/normal/*'):

    normal_images.append(mpimg.imread(img_path))



fig = plt.figure()

fig.suptitle('normal')

plt.imshow(normal_images[0], cmap='gray') 



covid_images = []

for img_path in glob.glob(DATASET_DIR + '/covid/*'):

    covid_images.append(mpimg.imread(img_path))



fig = plt.figure()

fig.suptitle('covid')

plt.imshow(covid_images[0], cmap='gray') 
print(len(normal_images))

print(len(covid_images))
IMG_W = 150

IMG_H = 150

CHANNELS = 3



INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)

NB_CLASSES = 2

EPOCHS = 48

BATCH_SIZE = 6
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(Conv2D(250,(3,3)))

model.add(Activation("relu"))

  

model.add(Conv2D(128,(3,3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2,2))

model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(AvgPool2D(2,2))



model.add(Conv2D(256,(2,2)))

model.add(Activation("relu"))

model.add(MaxPool2D(2,2))

    

model.add(Flatten())

model.add(Dense(32))

model.add(Dropout(0.25))

model.add(Dense(1))

model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



model.summary()
train_datagen = ImageDataGenerator(rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.3)



train_generator = train_datagen.flow_from_directory(

    DATASET_DIR,

    target_size=(IMG_H, IMG_W),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='training')



validation_generator = train_datagen.flow_from_directory(

    DATASET_DIR, 

    target_size=(IMG_H, IMG_W),

    batch_size=BATCH_SIZE,

    class_mode='binary',

    shuffle= False,

    subset='validation')



history = model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples // BATCH_SIZE,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples // BATCH_SIZE,

    epochs = EPOCHS)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
print("training_accuracy", history.history['accuracy'][-1])

print("validation_accuracy", history.history['val_accuracy'][-1])
label = validation_generator.classes
pred= model.predict(validation_generator)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (validation_generator.class_indices)

labels2 = dict((v,k) for k,v in labels.items())

predictions = [labels2[k] for k in predicted_class_indices]

print(predicted_class_indices)

print (labels)

print (predictions)
from sklearn.metrics import confusion_matrix



cf = confusion_matrix(predicted_class_indices,label)

cf
exp_series = pd.Series(label)

pred_series = pd.Series(predicted_class_indices)

pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames=['Predicted'],margins=True)
plt.matshow(cf)

plt.title('Confusion Matrix Plot')

plt.colorbar()

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show();
import cv2

def pred(path):

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    plt.imshow(image)

    IMG_SIZE=150

    array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    new_array= array.reshape(-1, IMG_SIZE, IMG_SIZE,3)

    prediction= model.predict(new_array)

    CATEGORIES=['covid','normal']

    label = str(CATEGORIES[int(prediction[0][0])])

    print(label)

pred('../input/atdataall/cn/3.jpeg')