import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from keras.preprocessing.image import ImageDataGenerator

from sklearn import metrics

from pathlib import Path
#Path 

train_dir = Path('../input/10-monkey-species/training/training/')

test_dir = Path('../input/10-monkey-species/validation/validation/')



#Images target size

img_width = 100

img_height = 100

target_size = (img_width, img_height)

channels = 3 #RGB
#Data augmentation 

train_generator = ImageDataGenerator(rescale=1/255,

                                    rotation_range=40,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,

                                    fill_mode='nearest')



valid_generator = ImageDataGenerator(rescale = 1/255)
epochs = 20

batch_size = 64 



#Finds images, transforms them

train_data = train_generator.flow_from_directory(train_dir, target_size=target_size, batch_size=batch_size,

                                                    class_mode='categorical')



test_data = valid_generator.flow_from_directory(test_dir, target_size=target_size, batch_size=batch_size,

                                                    class_mode='categorical', shuffle=False)
####################

# Model creation

####################



train_samples = train_data.samples

valid_samples = test_data.samples



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, channels)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(10, activation='softmax'))



#Compile model

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
#Fit the model

model.fit_generator(generator=train_data,

                    steps_per_epoch=train_samples/batch_size,

                    validation_data=test_data,

                    validation_steps=valid_samples/batch_size,

                    epochs=epochs)
#Get the true classes and the predictions for each image

test_steps_per_epoch = np.math.ceil(test_data.samples// batch_size+1)

predictions = model.predict_generator(test_data, steps=test_steps_per_epoch, verbose=1)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data.classes



#Get label names for the final table

cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']

labels = pd.read_csv("../input/10-monkey-species/monkey_labels.txt", names=cols, skiprows=1)

labels = labels['Common Name']



#Create classification report

print(metrics.classification_report(true_classes, predicted_classes,target_names=labels))
