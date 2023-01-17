import os

for folder in os.listdir('../input/natural-images/natural_images/'):

    print(folder)



# Any results you write to the current directory are saved as output.
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D, Dense, Input, Flatten, Dropout, BatchNormalization

from keras.models import Model, load_model

import tensorflow as tf

from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns 

import numpy as np

data_dir = '../input/natural-images/natural_images/'
BATCH_SIZE = 64

IMAGE_SIZE=(192,192)

IMAGE_SHAPE=(192,192,3)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   shear_range=0.2,

                                   zoom_range=0.3,

                                   horizontal_flip=True,

                                   validation_split=0.20) # set validation split

                                   

train_generator = train_datagen.flow_from_directory(

    data_dir,

    target_size = IMAGE_SIZE,

    shuffle=True,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    subset='training') # set as training data



validation_generator = train_datagen.flow_from_directory(

    data_dir, # same directory as training data

    target_size = IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle= False,

    class_mode='categorical',

    subset='validation') # set as validation data
resnet = ResNet50(include_top=False, weights='imagenet')

for layer in resnet.layers:

    layer.trainable = False
def classifiction_model():

    image_input = Input(shape=IMAGE_SHAPE)

    # Base feature extrction

    x = resnet(image_input)

    #x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.15)(x)

    x = BatchNormalization(axis=1)(x)

  

    x = Dense(512, activation='relu')(x)

    x = Dropout(0.25)(x)

    x = BatchNormalization(axis=1)(x)



    x = Flatten()(x)

    pred = Dense(8, activation='softmax')(x)

    return Model(inputs = image_input, outputs = pred)

    
model = classifiction_model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
filepath= "PersonDetector-{epoch:02d}-{val_accuracy:.4f}.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

callbacks_list = [checkpoint]
EPOCHS = 5

history = model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples//BATCH_SIZE,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples//BATCH_SIZE,

    epochs = EPOCHS,

    verbose=1,

    callbacks=callbacks_list)
#Confusion Matrix and Classification Report

validation_generator.reset

#Confution Matrix and Classification Report

Y_pred = model.predict_generator(validation_generator, validation_generator.samples // BATCH_SIZE+1)

y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')

print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')

target_names = ['fruit', 'cat', 'airplane', 'flower', 'person', 'car', 'dog', 'motorbike']

print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
ax= plt.subplot()

sns.heatmap(confusion_matrix(validation_generator.classes, y_pred), annot=True, ax = ax, fmt='g', cmap='Greens') #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 

ax.set_title('Confusion Matrix') 

ax.xaxis.set_ticklabels(['fruit', 'cat', 'airplane', 'flower', 'person', 'car', 'dog', 'motorbike']); ax.yaxis.set_ticklabels(['fruit', 'cat', 'airplane', 'flower', 'person', 'car', 'dog', 'motorbike'])