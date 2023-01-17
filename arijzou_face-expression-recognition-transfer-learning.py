%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import os 

import tensorflow as tf 

from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization

from keras.models import Sequential, Model

from keras.optimizers import RMSprop

from keras.applications import MobileNet

from keras.callbacks import ModelCheckpoint
os.chdir('../input/face-expression-recognition-dataset/images/')

train_path = './train/'

validation_path = './validation/'

os.listdir()
plt.figure(0, figsize=(48,48))

cpt = 0



for expression in os.listdir(train_path):

    for i in range(1,6):

        cpt = cpt + 1

        sp=plt.subplot(7,5,cpt)

        sp.axis('Off')

        img_path = train_path + expression + "/" +os.listdir(train_path + expression)[i]

        img = load_img( img_path, target_size=(48,48))

        plt.imshow(img, cmap="gray")





plt.show()
train_datagen = ImageDataGenerator(rescale=1/255,

                                   rotation_range=30,

                                   zoom_range=0.2)



validation_datagen = ImageDataGenerator(rescale=1/255)



batch_size = 128





train_generator = train_datagen.flow_from_directory (train_path,   

                                                     target_size=(48, 48),  

                                                     batch_size=batch_size,

                                                     shuffle=True,

                                                     class_mode='categorical',

                                                     color_mode="rgb")





validation_generator = validation_datagen.flow_from_directory(validation_path,  

                                                              target_size=(48,48), 

                                                              batch_size=batch_size,

                                                              class_mode='categorical',

                                                              color_mode="rgb")

#Loading the Mobilenet model 

featurizer = MobileNet(include_top=False, weights='imagenet', input_shape=(48,48,3))



#Since we have 7 types of expressions, we'll set the nulber of classes to 7

num_classes = 7



#Adding some layers to the feturizer

x = Flatten()(featurizer.output)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)

x = BatchNormalization()(x)

predictions = Dense(num_classes, activation = 'softmax')(x)







model = Model(input = featurizer.input, output = predictions)





model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
model_weights_path = r'/kaggle/working/model_weights.h5'



checkpoint = ModelCheckpoint(model_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')





history = model.fit(train_generator,

                        steps_per_epoch=train_generator.n//train_generator.batch_size,

                        validation_steps=validation_generator.n//validation_generator.batch_size,

                        epochs=60,

                        verbose=1,

                        validation_data = validation_generator,

                        callbacks=[checkpoint])
# serialize model to JSON

model_json = model.to_json()

with open(r'/kaggle/working/model.json', "w") as json_file:

    json_file.write(model_json)
plt.figure(figsize=(20,10))



plt.subplot(1, 2, 1)

plt.ylabel('Accuracy', fontsize=16)

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend(loc='upper right')





plt.subplot(1, 2, 2)

plt.ylabel('Loss', fontsize=16)

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend(loc='lower right')





plt.show()