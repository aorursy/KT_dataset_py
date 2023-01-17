%env CUDA_DEVICE_ORDER=PCI_BUS_ID

%env CUDA_VISIBLE_DEVICES=1
# Modules to ignore warnings

import warnings               

warnings.filterwarnings('ignore')



import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)



import os   

import shutil

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import call
from keras.models import Sequential, Model, load_model

from keras.layers import *

from keras.optimizers import Adam, SGD

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.applications.inception_v3 import InceptionV3

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.utils import to_categorical
#Mapping folders with respective classes 

class_map={'airplanes':'airplanes',

           'BACKGROUND_Google':'BACKGROUND_Google',

           'bonsai':'bonsai',

           'carside':'carside',

           'Faces':'Faces',

           'Faces_easy':'Faces_easy',

           'grand_piano':'grand_piano',

           'Leopards':'Leopards',

           'Motorbikes':'Motorbikes',

           'Watch':'Watch'}
train_path =  "../input/10_categories-1563192636507/10_categories"

classes = os.listdir(train_path)          # List of directories in train path

print(classes)
## Importing imageDataGenerator for image preprocessing/augumentation

## Augementaion gives images in all the angles

# It creates multiple copies of train images by (jittering)/Ading noise

#This includes rotating,shifting,zooming in,flipping 

datagen = ImageDataGenerator(

        rotation_range=30,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest') # 'nearest' is kind of algorithm to fill pixel values while transformation
# Create a directory named 'preview' to save augmented images.

aug_images_path = 'preview'

f = os.listdir(os.path.join(train_path,classes[7]))[0]

img_path = os.path.join(train_path,classes[3],f)

img = load_img(img_path)       # this is a PIL image

x = img_to_array(img)          # this is a Numpy array with shape (480, 640, 3)

x = x.reshape((1,) + x.shape)  

 

# Delete, if already exists

if os.path.isdir(aug_images_path):

#     os.system('rm -rf '+aug_images_path)

    shutil.rmtree(aug_images_path)

    

os.system('mkdir '+aug_images_path)

    

# the .flow() command below generates augmented images and saves them to a directory names 'preview'

i = 0

for batch in datagen.flow(x, batch_size=1, save_to_dir=aug_images_path, save_prefix='c0', save_format='jpg'):

    i += 1

    if i > 9:

        break
plt.imshow(img)

plt.axis('off')

plt.title('Original Image: '+f)
#plot the augumented images for the above original image

#Read them from 'preview' dictionary and display them

plt.figure(figsize=(20,6))

aug_images = os.listdir('preview')

for ix,i in enumerate(aug_images):

    img = mpimg.imread(os.path.join('preview',i))

    plt.subplot(2,5,ix+1)

    plt.imshow(img)

    plt.axis('off')

    plt.title(i)

    if ix==10:

        break 
# This is the augumentation configuration we will use for training

train_datagen = ImageDataGenerator(rescale=1/255.,validation_split=0.3,

                                    rotation_range=20,

                                    height_shift_range=0.2,

                                    zoom_range=0.2)



# This is a generator that will read pictures found in subfolers of 'train', and generates

# batches of augmented image data on the fly

train_generator = train_datagen.flow_from_directory(directory=train_path, 

                                                    batch_size=64, 

                                                    class_mode='categorical', 

                                                    shuffle=True, 

                                                    target_size=(299,299),subset="training")
# This is the augumentation configuration we will use for validation

val_datagen = ImageDataGenerator(rescale=1/255.)

val_generator = train_datagen.flow_from_directory(directory=train_path, 

                                                    batch_size=64, 

                                                    class_mode='categorical', 

                                                    shuffle=False, 

                                                    target_size=(299,299),subset="validation")
#We're using convolution layer to extract the features and building the model





def image_classifier(nb_classes):

    model = Sequential()



    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(299, 299, 3), padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dropout(0.2))



    model.add(Dense(128, init='uniform', activation='relu'))

    model.add(Dropout(0.4))



    model.add(Dense(nb_classes, activation='softmax'))

    

    return(model)
model = image_classifier(nb_classes=10)              ##compiling loss function and metrics to evaluate the model

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
## We're running the model on train and validation generator

hist0 = model.fit_generator(train_generator, 

                           validation_data=val_generator, 

                            validation_steps = 17,

                           epochs=5,steps_per_epoch=10).history
train_datagen = ImageDataGenerator(rescale=1/255.,validation_split=0.3,

                                    rotation_range=20,

                                    height_shift_range=0.2,

                                    zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(directory=train_path,subset='training',

                                                    batch_size=64, 

                                                    class_mode='categorical', 

                                                    shuffle=True, 

                                                    target_size=(299,299))
val_datagen = ImageDataGenerator(rescale=1/255.)

val_generator = train_datagen.flow_from_directory(directory=train_path,subset='validation',

                                                    batch_size=64, 

                                                    class_mode='categorical', 

                                                    shuffle=False, 

                                                    target_size=(299,299))
#Get inception architecture from keras.applications



from keras.applications.inception_v3 import InceptionV3



def inception_tl(nb_classes, freez_wts):

    

    trained_model = InceptionV3(include_top=False,weights='imagenet')

    x = trained_model.output

    x = GlobalAveragePooling2D()(x)

    pred_inception= Dense(nb_classes,activation='softmax')(x)

    model = Model(inputs=trained_model.input,outputs=pred_inception)

    

    for layer in trained_model.layers:

        layer.trainable=(1-freez_wts)

    

    return(model)
model = inception_tl(nb_classes=10, freez_wts=False)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

hist1 = model.fit_generator(train_generator, 

                           validation_data=val_generator, 

                          validation_steps = 17,

                           epochs=5,steps_per_epoch=10).history
## plotting the loss functions and accuracy

plt.figure(figsize=(8,3))

plt.subplot(1,2,1)

train_loss = plt.plot(hist1['loss'], label='train loss')

val_loss = plt.plot(hist1['val_loss'], label='val loss')

plt.legend()

plt.title('Loss')



plt.subplot(1,2,2)

train_loss = plt.plot(hist1['acc'], label='train acc')

val_loss = plt.plot(hist1['val_acc'], label='val acc')

plt.legend()

plt.title('Accuracy')
#We're building model using inception

model = inception_tl(nb_classes=10, freez_wts=False)

adam = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=adam)
nb_epochs = 10

init_lr = 0.001

min_lr = 0.000001

f = 10**(np.log10(min_lr/init_lr)/float(nb_epochs))



def poly_decay(epoch):

    ''' This function takes the current epoch as input and return the updated learning rate.

        The learning rate is multiplied by a factor 'f' after each epoch.

        In the first epoch, learning rate is set to 'init_lr'.

        By the end of 'nb_epochs' the learning rate is reduced to 'min_lr' '''

    return(init_lr*(f**epoch))



# ModelCheckpoint monitors the 'val_loss' and saves the model graph and weights at the epoch with least 'val_loss'

# 'save_weights_only'=True, saves only the weights

# 'save_weights_only'=False, saves the weights and the graph

chkp = ModelCheckpoint(filepath='inception_dd.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=0)

lr_schedule = LearningRateScheduler(poly_decay)
# plotting learning rate over Epoch's

lr_list = [poly_decay(i) for i in range(nb_epochs)]

plt.scatter(range(nb_epochs), np.log10(lr_list))

plt.plot(np.log10(lr_list))

plt.xlabel('Epoch')

plt.ylabel('Learning Rate in Log10 Scale')

plt.title('Learning Rate over Epochs')
## Building in the inception model

hist = model.fit_generator(train_generator, 

                           validation_data=val_generator, 

                           validation_steps = 17,

                           epochs=5,steps_per_epoch=10, 

                           callbacks=[chkp, lr_schedule]).history

np.savez('inception_dd_history.npz', loss=hist['loss'], acc=hist['acc'], val_loss=hist['val_loss'], val_acc=hist['val_acc'])
plt.figure(figsize=(8,3))

plt.subplot(1,2,1)

train_loss = plt.plot(hist['loss'], label='train loss')

val_loss = plt.plot(hist['val_loss'], label='val loss')

plt.legend()

plt.title('Loss')



plt.subplot(1,2,2)

train_loss = plt.plot(hist['acc'], label='train acc')

val_loss = plt.plot(hist['val_acc'], label='val acc')

plt.legend()

plt.title('Accuracy')
val_preds = model.predict_generator(generator=val_generator,steps=18)
val_preds_class = val_preds.argmax(axis=1)

val_preds_df = pd.DataFrame({'image':val_generator.filenames, 'prediction':val_preds_class})

val_preds_df.head(10)
submit['val_preds_df'] = predicted

submit.to_csv('submission.csv', index=False)

submit.head()