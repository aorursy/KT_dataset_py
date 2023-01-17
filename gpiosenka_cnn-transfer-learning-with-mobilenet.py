import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Activation,BatchNormalization

from tensorflow.keras.optimizers import  Adamax

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, load_model, Sequential

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import os

import cv2

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

logging.getLogger('tensorflow').setLevel(logging.FATAL)
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

logging.getLogger('tensorflow').setLevel(logging.FATAL)
train_dir=r'../input/100-bird-species/train'

test_dir=r'../input/100-bird-species/test'

val_dir=r'../input/100-bird-species/valid'
classes=os.listdir(test_dir) # class names are the names of the sub directories

class_count=len(classes)

batch_size=80

rand_seed=123

start_epoch=0 # specify starting epoch

epochs=15 # specify the number of epochs to run

img_size=224 # use 224 X 224 images compatible with mobilenet model

lr=.01 # specify initial learning rate

save_loc=r'./'  # specify directory where best model will be saved after training
def get_bs(dir,b_max):

    # dir is the directory containing the samples, b_max is maximum batch size to allow based on your memory capacity

    # you only want to go through test and validation set once per epoch this function determines needed batch size ans steps per epoch

    length=0

    dir_list=os.listdir(dir)

    for d in dir_list:

        d_path=os.path.join (dir,d)

        length=length + len(os.listdir(d_path))

    batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=b_max],reverse=True)[0]  

    return batch_size,int(length/batch_size)
b_max=80 # set maximum allowable batch size

test_batch_size, test_steps=get_bs(test_dir,b_max)

valid_batch_size, valid_steps=get_bs(val_dir, b_max)

train_batch_size=80

train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input, horizontal_flip=True).flow_from_directory(

        train_dir,  target_size=(img_size, img_size), batch_size=train_batch_size, seed=rand_seed, class_mode='categorical', color_mode='rgb')



valid_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) .flow_from_directory(val_dir, 

                    target_size=(img_size, img_size), batch_size=valid_batch_size,

                    class_mode='categorical',color_mode='rgb', shuffle=False)

test_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_dir,

                    target_size=(img_size, img_size), batch_size=test_batch_size,

                    class_mode='categorical',color_mode='rgb', shuffle=False )

test_file_names=test_gen.filenames  # save list of test files names to be used later

labels=test_gen.labels # save test labels to be used later

images,labels=next(train_gen)

plt.figure(figsize=(20, 20))

for i in range(25):

    plt.subplot(5, 5, i + 1)

    image=(images[i]+1 )/2

    plt.imshow(image)

    index=int(labels[i][1])

    plt.title(classes[index], color='black')

    plt.axis('off')

plt.show()
mobile = tf.keras.applications.mobilenet.MobileNet( include_top=False, input_shape=(img_size, img_size,3), pooling='max', weights='imagenet', dropout=.5) 

for layer in mobile.layers:

    layer.trainable=False

x=mobile.layers[-1].output # this is the last layer in the mobilenet model the global max pooling layer

x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)

predictions=Dense (len(classes), activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=predictions)    

#for layer in model.layers:

   # layer.trainable=True

model.compile(Adamax(lr=lr), loss='categorical_crossentropy', metrics=['accuracy']) 

#model.summary()
checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=save_loc, monitor='val_loss', verbose=1, save_best_only=True,

        save_weights_only=True, mode='auto', save_freq='epoch', options=None)

lr_adjust=tf.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.5, patience=1, verbose=1, mode="auto",

        min_delta=0.00001,  cooldown=0,  min_lr=0)

callbacks=[checkpoint, lr_adjust]
data=model.fit(x=train_gen,  epochs=epochs, verbose=1, callbacks=callbacks,  validation_data=valid_gen,  initial_epoch=start_epoch)
def tr_plot(results):

    tacc=results.history['accuracy']

    tloss=results.history['loss']

    vacc=results.history['val_accuracy']

    vloss=results.history['val_loss']

    Epoch_count=len(tloss)

    Epochs=[]

    for i in range (0,Epoch_count):

        Epochs.append(i+1)

    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss

    val_lowest=vloss[index_loss]

    index_acc=np.argmax(vacc)

    val_highest=vacc[index_acc]

    plt.style.use('fivethirtyeight')

    sc_label='best epoch= '+ str(index_loss+1)

    vc_label='best epoch= '+ str(index_acc + 1)

    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,5))

    axes[0].plot(Epochs,tloss, 'r', label='Training loss')

    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )

    axes[0].scatter(index_loss+1,val_lowest, s=150, c= 'blue', label=sc_label)

    axes[0].set_title('Training and Validation Loss')

    axes[0].set_xlabel('Epochs')

    axes[0].set_ylabel('Loss')

    axes[0].legend()

    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')

    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')

    axes[1].scatter(index_acc+1,val_highest, s=150, c= 'blue', label=vc_label)

    axes[1].set_title('Training and Validation Accuracy')

    axes[1].set_xlabel('Epochs')

    axes[1].set_ylabel('Accuracy')

    axes[1].legend()

    plt.tight_layout

    #plt.style.use('fivethirtyeight')

    plt.show()

tr_plot(data)
model.load_weights(save_loc)  

model.save(filepath=save_loc)
accuracy =100 *model.evaluate(test_gen, batch_size=test_batch_size, steps=test_steps)[1]

    
print('Model accuracy on Test Set is {0:7.2f} %'.format(accuracy))