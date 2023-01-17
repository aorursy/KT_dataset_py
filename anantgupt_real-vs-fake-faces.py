import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.layers import (Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, 

                          Dense, Flatten, Dropout)

from keras.optimizers import RMSprop, Adam, SGD

from keras import regularizers

from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
path = '../input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake'



def plot_img(path, set_):

    dir_ = os.path.join(path, 'train', set_)

    k = 0

    fig, ax = plt.subplots(3,3, figsize=(10,10))

    fig.suptitle(set_ + 'Faces')

    for j in range(3):

        for i in range(3):

            img = load_img(os.path.join(dir_, os.listdir(os.path.join(dir_))[k]))          

            ax[j,i].imshow(img)

            ax[j,i].set_title("")

            ax[j,i].axis('off')

            k +=1

  #  fig.tight_layout()

    plt.suptitle(set_ + ' Faces')

    return plt



plot_img(path, 'real').show()
plot_img(path, 'fake').show()
bs = 64

row, col = 224, 224

train_datagen = ImageDataGenerator(rescale=1./255,

                                   horizontal_flip=True

                                  )

training_set = train_datagen.flow_from_directory(path + '/train',

                                                 class_mode='binary',

                                                 shuffle=True,

                                                 target_size=(row,col),

                                                 batch_size=bs

                                                )

val_test_datagen = ImageDataGenerator(rescale=1./255)



validation_set = val_test_datagen.flow_from_directory(path + '/valid',

                                                      class_mode='binary',

                                                      shuffle=True,

                                                      target_size=(row,col),

                                                      batch_size=bs

                                                     ) 

test_set = val_test_datagen.flow_from_directory(path + '/test',

                                                class_mode='binary',

                                                shuffle=True,

                                                target_size=(row,col),

                                                batch_size=bs

                                               )

training_set.class_indices
from tensorflow.keras.applications import DenseNet121

def get_model():

    densenet = DenseNet121(weights='imagenet',

                           include_top=False,

                           input_shape=(224,224,3)

                          )

    model = tf.keras.models.Sequential([densenet,

                                        GlobalAveragePooling2D(),

                                        Dense(512, activation='relu'),

                                        BatchNormalization(),

                                        Dropout(0.3),

                                        Dense(1, activation='sigmoid')

                                      ])

    model.compile(optimizer=Adam(lr=0.001),

                loss='binary_crossentropy',

                metrics=['accuracy']

                )



    return model



spoofnet = get_model()

spoofnet.summary()
checkpoint = ModelCheckpoint(filepath='spoffnet.h5',

                             save_best_only=True,

                             verbose=1,

                             mode='min',

                             moniter='val_loss'

                            )

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 

                              factor=0.2, 

                              patience=3, 

                              verbose=1, 

                              min_delta=0.0001

                             )

csv_logger = CSVLogger('training.log')



callbacks = [checkpoint, reduce_lr, csv_logger]
steps_per_epoch = training_set.n // training_set.batch_size

validation_steps = validation_set.n // validation_set.batch_size



hist = spoofnet.fit(training_set,

                    validation_data=validation_set,

                    callbacks=callbacks,

                    steps_per_epoch=steps_per_epoch,

                    validation_steps=validation_steps,

                    epochs=4

                   )
plt.figure(figsize=(14,5))

plt.subplot(1,2,2)

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['train', 'val'])



plt.subplot(1,2,1)

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['train', 'val'])

plt.show()
_, accu = spoofnet.evaluate(test_set)

print('Final Test Acccuracy = {:.3f}'.format(accu*100))
y_pred = spoofnet.predict(test_set)

y_pred = (y_pred < 0.5).astype(np.int)



from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(test_set.classes, y_pred)

print('Confusion Matrix')

print(cm)



names = ['fake', 'real']

print('Classification Report')

print(classification_report(test_set.classes, y_pred, target_names=names))



plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.colorbar()

tick_mark = np.arange(len(names))

_ = plt.xticks(tick_mark, names, rotation=90)

_ = plt.yticks(tick_mark, names)