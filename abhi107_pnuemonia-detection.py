from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K



import os

import numpy as np

import pandas as pd

import pathlib

import matplotlib.pyplot as plt

%matplotlib inline
print(os.listdir("../input/chest-xray-pneumonia/chest_xray"))



print(os.listdir("../input/chest-xray-pneumonia/chest_xray/train"))

TRAIN_PATH = "../input/chest-xray-pneumonia/chest_xray/train/"

train_dir = pathlib.Path(TRAIN_PATH)

print(os.listdir("../input/chest-xray-pneumonia/chest_xray/test/"))

TEST_PATH  = "../input/chest-xray-pneumonia/chest_xray/test/"

test_dir =pathlib.Path(TEST_PATH)

print(os.listdir("../input/chest-xray-pneumonia/chest_xray/val/"))

VAL_PATH = "../input/chest-xray-pneumonia/chest_xray/val/"

val_dir = pathlib.Path(VAL_PATH)
img_name = 'person1000_virus_1681.jpeg'

img_normal = load_img(TRAIN_PATH+"PNEUMONIA/" + img_name)



print('PNUEMONIA')

plt.imshow(img_normal)

plt.show()
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])

CLASS_NAMES
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    validation_split=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    rotation_range=10,

    )

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(226,226),

    batch_size=32,

    shuffle=True,

    class_mode='binary',

    subset='training'

  )

validation_generator = train_datagen.flow_from_directory(train_dir,target_size=(226,226),batch_size=32,  class_mode='binary',classes = list(CLASS_NAMES),subset='validation')
def show_batch(image_batch, label_batch):

  plt.figure(figsize=(10,10))

  for n in range(25):

      ax = plt.subplot(5,5,n+1)

      plt.imshow(image_batch[n])

      plt.title(CLASS_NAMES[int(label_batch[n])])

      plt.axis('off')
image_batch, label_batch = validation_generator[6]

show_batch(image_batch, label_batch)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(226,226,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])


IDG_test = ImageDataGenerator(rescale = 1./255)

test_data = IDG_test.flow_from_directory(test_dir,target_size=(226,226),shuffle=False,batch_size=32,class_mode='binary',classes = list(CLASS_NAMES))
history =model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples //32,

    epochs=7,

    validation_data=validation_generator,

   )
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.evaluate(test_data)
import math

test_labels = []

number_of_examples = len(test_data.filenames)

number_of_generator_calls = math.ceil(number_of_examples / (1.0 * 32))



for i in range(0,int(number_of_generator_calls)):

    test_labels.extend(np.array(test_data[i][1],dtype=int))
test_labels = np.array(test_labels)
image_batch, label_batch = test_data[1]

preds = model.predict(test_data)



for i in range(len(preds)):

    if(preds[i]>=.5):

        preds[i]=1

    

    else:

         preds[i]=0

    

preds = preds.ravel()   

preds = np.array(preds,dtype=int)



from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix



cm  = confusion_matrix(test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.2f}".format(precision))