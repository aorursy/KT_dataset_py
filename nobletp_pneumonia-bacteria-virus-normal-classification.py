# import libraries

import json

import math

import os

from glob import glob 

from tqdm import tqdm

from PIL import Image

import cv2 # image processing

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # data visualization





from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report 

from sklearn.model_selection import train_test_split



from keras import layers

from keras.models import Sequential

from keras.optimizers import Adam, RMSprop

from keras.applications import VGG16, ResNet50, InceptionResNetV2, InceptionV3

from keras.utils.np_utils import to_categorical

from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator,image,img_to_array,load_img

import tensorflow as tf

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam,RMSprop,SGD
import timeit



device_name = tf.test.gpu_device_name()

if "GPU" not in device_name:

    print("GPU device not found")

print('Found GPU at: {}'.format(device_name))
input_dir = "../input/xray-cat/"

train_dir = input_dir +"train/"

test_dir = input_dir +"test/"

#val_dir = input_dir +"val/"
import glob

def return_df(path1, path2):

    

    images1 = glob.glob(path1)

    images2 = glob.glob(path2)

    

    image_dict = {}

    images = images1+ images2

    

    for i in range(len(images)):

        try:

            if images[i].split('/')[-1].split('_')[1] == 'bacteria':

                image_dict[images[i]] = 'bacteria'

            if images[i].split('/')[-1].split('_')[1] == 'virus':

                image_dict[images[i]] = 'virus'

        except:

            image_dict[images[i]] = 'normal'



    image_df = pd.DataFrame.from_dict(image_dict,orient = 'index' ).reset_index()

    image_df.rename(columns = {'index':'path',0:'status'}, inplace = True)

    

    return image_df

    
#path1 = "../input/chest-xray-pneumonia/chest_xray/train/NORMAL/*.jpeg"

#path2 = "../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/*.jpeg"



#df_train = return_df(path1,path2)
#df_train.head()
#df_train.status.value_counts()
#path1 = "../input/chest-xray-pneumonia/chest_xray/test/NORMAL/*.jpeg"

#path2 = "../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/*.jpeg"



#df_test = return_df(path1,path2)
#df_test.head()
#df_test.status.value_counts()
def process_data(img_dims, batch_size):

    # Data generation objects

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,

                                       zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

    

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    

    # This is fed to the network in the specified batch sizes and image dimensions

    train_gen = train_datagen.flow_from_directory(

    directory=train_dir, 

    target_size=(img_dims, img_dims), 

    batch_size=batch_size, 

    class_mode='categorical')



    test_gen = test_val_datagen.flow_from_directory(

    directory=test_dir, 

    target_size=(img_dims, img_dims), 

    batch_size=batch_size, 

    class_mode='categorical')

    

    # I will be making predictions off of the test set in one batch size

    # This is useful to be able to get the confusion matrix

    test_data = []

    test_labels = []



   

    for cond in ['/normal/', '/bacteria/', '/virus/']:

        for img in (os.listdir(test_dir + cond)):

            img = plt.imread(test_dir+cond+img)

            img = cv2.resize(img, (img_dims, img_dims))

            img = np.dstack([img, img, img])

            img = img.astype('float32') / 255

            if cond=='/normal/':

                label = 0

            elif cond=='/bacteria/':

                label = 1

            elif cond=='/virus/':

                label = 2

            test_data.append(img)

            test_labels.append(label)

        

    test_data = np.array(test_data)

    test_labels = np.array(test_labels)

    

    return train_gen, test_gen, test_data, test_labels


#def process_data(img_dims, batch_size):

#    # Data generation objects

#    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,

#                                       zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

#    

#    test_val_datagen = ImageDataGenerator(rescale=1./255)

#    

#    # This is fed to the network in the specified batch sizes and image dimensions

##    train_gen = train_datagen.flow_from_directory(

#    directory=train_dir, 

#    target_size=(img_dims, img_dims), 

#    batch_size=batch_size, 

#    class_mode='binary')

#

#    test_gen = test_val_datagen.flow_from_directory(

#    directory=test_dir, 

#    target_size=(img_dims, img_dims), 

#    batch_size=batch_size, 

#    class_mode='binary')

#    

#    # I will be making predictions off of the test set in one batch size

#    # This is useful to be able to get the confusion matrix

#    test_data = []

#    test_labels = []

#

#    for cond in ['/NORMAL/', '/PNEUMONIA/']:

#        for img in (os.listdir(test_dir + cond)):

#            img = plt.imread(test_dir+cond+img)

#            img = cv2.resize(img, (img_dims, img_dims))

#            img = np.dstack([img, img, img])

#            img = img.astype('float32') / 255

#            if cond=='/NORMAL/':

#                label = 0

#            elif cond=='/PNEUMONIA/':

#                label = 1

#            test_data.append(img)

#            test_labels.append(label)

#        

#    test_data = np.array(test_data)

#    test_labels = np.array(test_labels)

#    

#    return train_gen, test_gen, test_data, test_labels

img_dims = 150

batch_size = 60

train_gen, test_gen, test_data, test_labels = process_data(img_dims, batch_size)
# Data Augmentation

train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='categorical',

        classes = ['normal','bacteria','virus'])

test_generator = test_datagen.flow_from_directory(

        test_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='categorical',

        classes = ['normal','bacteria','virus'])

# validation_generator = test_datagen.flow_from_directory(

#         val_dir,

#         target_size=(150, 150),

#         batch_size=20,

#         class_mode='binary')
from keras.optimizers import Adam, RMSprop, Adamax
# Create ResNet50 Model with Keras library



#adamax = tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax", **kwargs)



# Create vggnet Model with Keras library

vgg16 = VGG16(

    weights='imagenet',

    include_top=False,

    input_shape=(150,150,3)

)



ResNet_weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

Inception_ResNet_weights = '../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

Inception_weights = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



#resnet_50 = ResNet50(weights = 'imagenet', include_top = False, input_shape = (150,150,3))

#inception_resnet_v2 = InceptionResNetV2(weights= Inception_ResNet_weights, include_top=False, input_shape=(150, 150, 3))

inception_v3 = InceptionV3(weights= Inception_weights, include_top=False, input_shape=(150, 150, 3))



decay = 1e-4/20

sgd = SGD(lr=1e-4, momentum=0.9, decay=decay, nesterov=False)



def build_model(backbone, lr=1e-4):

    model = Sequential()

    model.add(backbone)

    model.add(BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())

    model.add(Dense(1024, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(3,activation="sigmoid"))

    

    model.compile(

        loss='categorical_crossentropy',

        

        optimizer=sgd, #Adamax(lr=lr),#'rmsprop',#Adamax(lr=lr),

        metrics= [tf.keras.metrics.CategoricalAccuracy()]#['acc' ]

    )

    return model



#model = build_model(vgg16 ,lr = 1e-4)

#model = build_model(ResNet_weights,lr = 1e-4)

#model = build_model(inception_resnet_v2, lr = 1e-4)

model = build_model(inception_v3, lr = 1e-4)

model.summary()
images_normal = len(os.listdir('../input/xray-cat/train/normal'))

images_bacteria = len(os.listdir('../input/xray-cat/train/bacteria'))

images_virus = len(os.listdir('../input/xray-cat/train/virus'))



total_images = images_normal + images_bacteria + images_virus
images_bacteria
weight_for_0 = (1 / images_normal)*(total_images)/2.0 

weight_for_1 = (1 / images_bacteria)*(total_images)/2.0

weight_for_2 = (1 / images_virus)*(total_images)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))

print('Weight for class 2: {:.2f}'.format(weight_for_2))
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=3)



# Fit Model

history2 = model.fit_generator(

    train_generator,

    steps_per_epoch=200,

    epochs=30,

    validation_data=test_generator,

    validation_steps=10,

    callbacks=[early_stopping_monitor],

    class_weight = {0:0.5,1:1.5,2:1.5}

)
# Visualize Loss and Accuracy Rates

fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax = ax.ravel()



for i, met in enumerate(['categorical_accuracy', 'loss']):

    ax[i].plot(history2.history[met])

    ax[i].plot(history2.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import multilabel_confusion_matrix
results = model.evaluate_generator(test_generator)

preds   = model.predict_generator(test_generator)

#print('The current model achieved a categorical accuracy of {}%!'.format(round(results[0]*100,2)))

print('The current model achieved a categorical accuracy of {}%!'.format(round(results[1]*100,2)))

y_pred = np.argmax(preds, axis=1)
print('Confusion Matrix')

print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')

target_names = ['normal', 'bacteria', 'virus']

print(classification_report(test_generator.classes, y_pred, target_names=target_names))