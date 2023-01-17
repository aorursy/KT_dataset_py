import os

from sklearn.metrics import confusion_matrix

import cv2

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential, load_model

from keras.callbacks import ModelCheckpoint

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import random,os,glob

import numpy as np

import time

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Conv2D,Dense, Dropout,MaxPooling2D,GlobalAveragePooling2D

from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam
data_path = "../input/pavbhaji/"
strategy = tf.distribute.get_strategy()



print(strategy.num_replicas_in_sync)
train_datagen = ImageDataGenerator(

        rescale = 1./255,

        rotation_range = 20,

        width_shift_range = 0.2,

        height_shift_range = 0.2,

        horizontal_flip = True,

        vertical_flip = True,

        fill_mode='nearest'

)

validation_datagen = ImageDataGenerator(

        rescale = 1./255

)

test_datagen = ImageDataGenerator(

        rescale = 1./255

)
img_shape = (224, 224, 3) # default values

train_batch_size = 64

val_batch_size = 8
train_generator = train_datagen.flow_from_directory(

            data_path + '/train', 

            target_size = (img_shape[0], img_shape[1]),

            batch_size = train_batch_size,

            class_mode = 'categorical',

            color_mode="rgb",

            shuffle = True) #binary - not working



validation_generator = validation_datagen.flow_from_directory(

            data_path + '/valid',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = val_batch_size,

            class_mode = 'categorical',

            color_mode="rgb",

            shuffle = False) #False



test_generator = test_datagen.flow_from_directory(

            data_path + '/test',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = val_batch_size,

            class_mode = 'categorical',

            color_mode="rgb",

            shuffle = False)
labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:

  f.write(labels)

labels = dict((v,k) for k,v in train_generator.class_indices.items())

print(labels)
from tensorflow.keras.applications import InceptionV3

inception = InceptionV3(weights = 'imagenet',include_top = False, input_shape = img_shape)
for layer in inception.layers[:-50]:

    layer.trainable = False
with strategy.scope():

    

# Create the model

    model = Sequential()



     # Add the convolutional base model

    model.add(inception)

    model.add(Conv2D(128, 3, activation='relu'))

    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

    # Add new layers

    model.add(Flatten())



    model.add(Dense(512, activation='relu'))

    #model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))

    # last layer

    model.add(Dense(2, activation='softmax')) #relu,sigmoid #not 1 -categorical
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

mc = ModelCheckpoint('inception.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True) #Inception.h5



steps_per_epoch = train_generator.samples//train_generator.batch_size

validation_steps = validation_generator.samples//validation_generator.batch_size

start = time.time()

history = model.fit_generator(

    train_generator,

    steps_per_epoch=steps_per_epoch ,

    epochs=50,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1,

    workers=4,

    callbacks=[es, mc])

end = time.time()

print('Execution time: ', end-start)
# summarize history for accuracy

plt.plot(history.history['accuracy'] )

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
test_steps = test_generator.samples//test_generator.batch_size

test_generator.reset()

prediction = model.predict_generator(test_generator,steps = test_steps,verbose=1)

pred_binary = [np.argmax(value[0]) for value in prediction] 

pred_binary = np.array(pred_binary)



import collections

collections.Counter(pred_binary)



import pandas as pd



Id = os.listdir("%s/test/PavBhaji"%data_path)

Id.extend(os.listdir("%s/test/Not_PavBhaji"%data_path))

pred_list_new = [labels[f] for f in pred_binary]



test_df = pd.DataFrame({'Image_name': Id,'Predicted_Label': pred_list_new})

test_df.to_csv('submission.csv', header=True, index=False)

test_df
def load_img(img_path):

    from keras.preprocessing import image

    img = image.load_img(img_path, target_size=(224, 224))

    img = image.img_to_array(img, dtype=np.uint8)

    img = np.array(img)/255.0

    plt.axis('off')

    plt.imshow(img.squeeze())

    return(img)
def prediction(img):

    

     model = tf.keras.models.load_model("inception.h5")

     p = model.predict(img[np.newaxis, ...])

     classes=[]

     prob=[]

     print("\nPROBABILITY OF EACH IMAGE\n")

     for i,j in enumerate (p[0],0):

         print(labels[i].upper(),':',round(j*100,2),'%')

         classes.append(labels[i])

         prob.append(round(j*100,2))

         

     def plot_bar_x():



         index = np.arange(len(classes))

         plt.bar(index, prob)

         plt.xlabel('Labels', fontsize=12)

         plt.ylabel('Probability', fontsize=12)

         plt.xticks(index, classes, fontsize=12, rotation=20)

         plt.title('Probability for loaded image')

         plt.show()

     plot_bar_x()
img=load_img(data_path +'/test/Not_PavBhaji/39744547_231289940899288_5133092913563041792_n.jpg')
prediction(img)
img=load_img(data_path +'/test/PavBhaji/Pav-Bhaji-3.jpg')
prediction(img)
img=load_img(data_path +'/test/Not_PavBhaji/39979218_907398179465424_4280824617234333696_n.jpg')
prediction(img)
img=load_img(data_path +'/test/PavBhaji/pav-bhaji-recipe.jpg')
prediction(img)