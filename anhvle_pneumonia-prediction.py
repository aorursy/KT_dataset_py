# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.image  as mpimg

import matplotlib.pyplot as plt

import os

from PIL import Image

from sklearn.metrics import classification_report,confusion_matrix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
IMAGE_SIZE = 64

TRAIN_DIR = '../input/chest-xray-pneumonia/chest_xray/train/'

TEST_DIR = '../input/chest-xray-pneumonia/chest_xray/test/'

VALIDATION_DIR = '../input/chest-xray-pneumonia/chest_xray/val/'

BATCH_SIZE = 100

DESIRED_ACCURACY = 0.99

# STEPS_PER_EPOCH = int(5000 / BATCH_SIZE)
# normalise data to 0 - 1

train_datagen = ImageDataGenerator(rescale=1.0/255.,

                                   zoom_range=0.2,

                                  # horizontal_flip=True,

                                   width_shift_range=0.1,

                                   height_shift_range=0.1,

                                  )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,

                                                    batch_size=BATCH_SIZE,

                                                    class_mode='binary',

                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE))



# normalise data to 0 - 1

test_datagen = ImageDataGenerator(rescale=1.0/255.)

test_generator = test_datagen.flow_from_directory(TEST_DIR,

                                                    batch_size=BATCH_SIZE,

                                                    class_mode='binary',

                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE))



# normalise data to 0 - 1

validation_datagen = ImageDataGenerator(rescale=1.0/255.)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,

                                                    batch_size=BATCH_SIZE,

                                                    class_mode='binary',

                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE))
#Normal pic 

norm_pic_address = TRAIN_DIR + 'NORMAL/IM-0125-0001.jpeg'



#Pneumonia

sic_address = TRAIN_DIR + 'PNEUMONIA/person1000_bacteria_2931.jpeg'



# Load the images

norm_load = Image.open(norm_pic_address)

sic_load = Image.open(sic_address)



#Let's plt these images

f = plt.figure(figsize= (10,6))

a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(norm_load)

a1.set_title('Normal')



a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(sic_load)

a2.set_title('Pneumonia')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), padding='same'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),

    tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),

#     tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()
# callback to stop if good accuracy

class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs=None):

    if(logs.get('acc')>DESIRED_ACCURACY):

      print("Reached " + str(DESIRED_ACCURACY * 100) + "% accuracy so cancelling training!")

      self.model.stop_training = True



callbacks = [myCallback()]
history = model.fit(train_generator,

                    epochs=10,

                    # steps_per_epoch=STEPS_PER_EPOCH,

                    validation_data=validation_generator,

                    # validation_steps=8,

                    # callbacks=callbacks,

                    verbose = 1

                   )
test_history = model.evaluate_generator(test_generator)

print("The testing loss is : " , test_history[0]*100 , "%")

print('The testing accuracy is :',test_history[1]*100, '%')
predicts = model.predict(test_generator)

predictions = [round(x[0]) for x in predicts.tolist()]

truths = list(test_generator.classes)
print(classification_report(truths, predictions, target_names = ['Pneumonia (Class 1)','Normal (Class 0)']))
cm = confusion_matrix(truths, predictions)
# Accuracy 

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()



# Loss 



plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
# Get random images

norm_ind = np.random.randint(0, 7)

sick_ind = np.random.randint(0, 7)

norm = TEST_DIR + test_generator.filenames[norm_ind]

sick = TEST_DIR + test_generator.filenames[sick_ind]



# Load the images

norm_load = Image.open(norm)

sic_load = Image.open(sick)





print(test_generator.class_indices)

#Let's plt these images

f = plt.figure(figsize= (10,6))



p1 = model.predict(test_generator[norm_ind])

a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(norm_load)

a1.set_title(str(p1[0]) + ' (' + test_generator.filenames[norm_ind] + ')')



p2 = model.predict(test_generator[sick_ind])

a2 = f.add_subplot(1, 2, 2)

img_plot = plt.imshow(sic_load)

a2.set_title(str(p2[0]) + ' (' + test_generator.filenames[sick_ind] + ')')