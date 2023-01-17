import tensorflow as tf
from pathlib import Path
from PIL import Image
import IPython
from pprint import pprint
from urllib import request
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
plt.ion()
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
tf.version.VERSION
DATA_DIR="../input/"
MODEL_DIR="../output/kaggle/working/model/"
FRUITS_DATASET = DATA_DIR+'/fruits/fruits-360/'
model = None

CLASS_NAMES = []
IMAGE_SIZE=[100, 100]
BATCH_SIZE=8

if tf.version.VERSION == '2.3.0':
    training_ds, validation_ds = (
                    tf.keras.preprocessing.image_dataset_from_directory(
                            shuffle=True, directory=FRUITS_DATASET+'Training/', 
                            label_mode='int', batch_size=BATCH_SIZE, color_mode='rgb',
                            image_size=IMAGE_SIZE,
                            seed=6, subset='training', 
                            validation_split=.2,
                    ),
                    tf.keras.preprocessing.image_dataset_from_directory(
                            shuffle=True, directory=FRUITS_DATASET+'Training/', 
                            label_mode='int', batch_size=BATCH_SIZE, color_mode='rgb',
                            image_size=IMAGE_SIZE,
                            seed=6, subset='validation', 
                            validation_split=.2,
                    )
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                        shuffle=False, directory=FRUITS_DATASET+'Test/', 
                        label_mode='int', batch_size=BATCH_SIZE, color_mode='rgb',
                        image_size=IMAGE_SIZE,
             )
    training = training_ds.map(lambda x,y: (x/255., y))
    validation = validation_ds.map(lambda x,y: (x/255., y))
    test = test_ds.map(lambda x,y: (x/255., y))    
   
    CLASS_NAMES = training_ds.class_names
    
else:    
    train_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                                rescale=1./255, 
                                data_format='channels_last', 
                                validation_split=0.2, 
                        )    
    training_ds = train_image_gen.flow_from_directory(
                                FRUITS_DATASET+'Training/', 
                                subset='training', 
                                seed=6, class_mode='sparse',
                                batch_size=BATCH_SIZE, 
                                shuffle=True, target_size=IMAGE_SIZE
                    )
    validation_ds = train_image_gen.flow_from_directory(
                                        FRUITS_DATASET+'Training/', 
                                        class_mode='sparse', 
                                        batch_size=BATCH_SIZE,
                                        shuffle=True, 
                                        target_size=IMAGE_SIZE,
                                        subset='validation', 
                                        seed=6,                                         
                    )    
    test_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rescale=1./255, 
                                    data_format='channels_last'
                        )
    
    test_ds = test_image_gen.flow_from_directory(
                                                FRUITS_DATASET+'Test/', 
                                                class_mode='sparse', 
                                                batch_size=BATCH_SIZE,shuffle=False, 
                                                target_size=IMAGE_SIZE
                    )
     
    CLASS_NAMES = training_ds.class_indices
fig, ax = plt.subplots(1, 8, figsize=(16,16))

for batch in training:
    image_batch, label_batch = batch
    for iindex in range(8): 
        ax[iindex].grid(False)
        ax[iindex].set_title(CLASS_NAMES[label_batch[iindex].numpy()])
        ax[iindex].imshow(image_batch[iindex])
        ax[iindex].axis('off')
    break
import time
if False:
    model = tf.keras.models.load_model(filepath='model/FruDetec/')
else:
    now = time.time()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (5,5), activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Conv2D(16, (5,5),activation='relu'),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(192, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4), 
                  metrics=['accuracy'])
    
    model.build((None, IMAGE_SIZE[0], IMAGE_SIZE[1],3))
    model.summary()
    
    model.fit(training, epochs=10, 
              validation_data=validation, use_multiprocessing=True,
              callbacks=[tf.keras.callbacks.EarlyStopping()])
    
    # save model
    model.save(filepath='model/FruDetec', overwrite=True)
    print("{}s".format(time.time()-now))
model_layers = tf.keras.models.Model(inputs=model.inputs, outputs=[l.output for l in model.layers])
from matplotlib import pyplot as plot
plot.ion()
%matplotlib inline

import random

f, ax= plt.subplots(4, 4, figsize=(8,8))

channels = random.sample(range(0, 16), 4)
layers = range(0, 4)

test_shuffled = test.unbatch().shuffle(seed=6, buffer_size=1024).batch(BATCH_SIZE)

for ibatch in test_shuffled:  
    image_batch, label_batch = ibatch
    outputs = model_layers.predict(ibatch) ##(layers, batches, nrows, ncols, channel) 
    
    for i in range(0, 4):  
        class_int = tf.argmax(outputs[len(outputs)-1][i], axis=0).numpy()
        # IPython.display.display(tf.keras.preprocessing.image.array_to_img(image_batch[i]))
        print("T={},P={}".format(CLASS_NAMES[label_batch[i]], CLASS_NAMES[class_int]))
        
        for c, l in zip(channels, layers):
            #print(len(outputs))                
            # print(outputs[0][i].shape)            
            # print(outputs[1][i].shape)            
            # print(outputs[2][i].shape)            
            # print(outputs[3][i].shape)            
            ax[i, l].set_title("I{}, L{}, C{}".format(i, l, c))
            ax[i, l].grid(False)
            ax[i, l].set_xticks([])
            ax[i, l].set_yticks([])
            ax[i, l].imshow(outputs[l][i, :,:, c], cmap='inferno')
            ax[i, l].axis('off')
    break
y_pred=model.predict(test)
y_pred=np.argmax(y_pred, 1)
y_pred.shape
y_true = test.unbatch().map(lambda x, y: y)
y_true = np.array(list(y_true.as_numpy_iterator()))
y_true.shape
score = y_pred == y_true

"Total-{}, Hit-{}, Miss-{}".format(len(score), len(score[y_pred == y_true]), len(score[y_pred != y_true]))

