import random, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
target_names = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'brandine_spuckler','carl_carlson', 
                  'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler','comic_book_guy', 'disco_stu', 'dolph_starbeam','duff_man','edna_krabappel', 
                  'fat_tony', 'gary_chalmers','gil', 'groundskeeper_willie', 'homer_simpson','jimbo_jones', 'kearney_zzyzwicz','kent_brockman', 'krusty_the_clown', 'lenny_leonard', 
                  'lionel_hutz', 'lisa_simpson', 'lunchlady_doris','maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby','milhouse_van_houten', 
                  'miss_hoover', 'moe_szyslak', 'ned_flanders','nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner', 
                  'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum','selma_bouvier', 'sideshow_bob', 'sideshow_mel', 
                  'snake_jailbird','troy_mcclure', 'waylon_smithers']


nb_classes = len(target_names)
train_data_path ='../input/the-simpsons-dataset-compilation-49-characters/simpsons_dataset/simpsons_dataset'
test_data_path = '../input/the-simpsons-dataset-compilation-49-characters/simpsons_testdataset/simpsons_testdataset'
path = test_data_path 
imsize = [64,64]
sz2=imsize[0]
sz1=imsize[1]

img_rows = imsize[0]
img_cols = imsize[1]
epochs = 200
batch_size = 128
num_of_train_samples = 29894
num_of_test_samples = 4555



#Image Generator (It would probably not be needed, but used to improve)
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True, # randomly flip images
                                   fill_mode='nearest')
    
train_generator=train_datagen.flow_from_directory(train_data_path,
                                                 target_size=(img_rows,img_cols),
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(img_rows,img_cols,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), padding='same')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#########################################################
#Set model CheckPoints to plot later the evolution of accuracy and loss
#########################################################

checkpoint = ModelCheckpoint('model_weightsk.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
history=model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size,
                    callbacks=callbacks_list)
model_json = model.to_json()
with open('modelk.json', 'w') as json_file:
    json_file.write(model_json)
model.save('model_weightsk.h5')  
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()