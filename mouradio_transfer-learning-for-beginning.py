# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



import os, zipfile, random



# Import the backend

import tensorflow as tf



# Data preprocessing

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical



# Model architecture

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



# Model evaluation

from sklearn.metrics import accuracy_score

import os, cv2, re, random
img_size = (150, 150)

input_dim = (150, 150, 3)

epochs = 30

batch_size = 16
local_zip_train = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'

local_zip_test = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

zip_ref = zipfile.ZipFile(local_zip_train, 'r')

zip_ref.extractall('/kaggle/working')

zip_ref.close()





local_zip_train = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'

local_zip_test = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

zip_ref = zipfile.ZipFile(local_zip_train, 'r')

zip_ref.extractall('/kaggle/working')

zip_ref.close()







filenames = os.listdir("/kaggle/working/train")

dogs = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        dogs.append(1)

    else:

        dogs.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'dogs': dogs

})



df.head()
images_list = []

for i in df['filename']:

    i='/kaggle/working/train/'+i

    images_list.append(i)
img_width = 150

img_height = 150

def prepare_data(list_of_images):

    """

    Returns two arrays: 

        x is an array of resized images

        y is an array of labels

    """

    x = [] # images as arrays

    y = np.array(df['dogs']) # labels

   # y1= []

   # prepared_images=[]

    """  for i in list_of_images:

        image = load_img(i, target_size=(150, 150))

        image = img_to_array(image)

        image = image.reshape((150,150, 3))

        x.append(preprocess_input(image)) """

    for image in list_of_images:

        x.append(cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))

        

  #  for i in x:

    #    prepared_images.append(preprocess_input(i))

    

    return x, y

    



from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

from keras.layers import Dense, Flatten, Dropout, Lambda, Input, Concatenate, concatenate

X,Y = prepare_data(images_list) 
import random as rn



fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(Y))

        ax[i,j].imshow(X[l])

        if Y[l]==0 :

            ax[i,j].set_title(' Cat ')

        else :

            ax[i,j].set_title(' Dog ')

        

        

plt.tight_layout()
X_train, X_val, Y_train, Y_val = train_test_split(np.array(X),Y, test_size=0.2, random_state=1)
import keras

from keras import backend

from keras.applications.inception_v3 import InceptionV3,preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD

from keras.models import Model, load_model

from keras.layers import Dense, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

#Prepare call backs

LR_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=2, factor=.5, min_lr=.00001)

EarlyStop_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

my_callback=[EarlyStop_callback, LR_callback]
InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)

x = InceptionV3_base_model.output

x_pool = GlobalAveragePooling2D()(x)

x_poole = Dropout(0.4)(x_pool)

final_predect = Dense(1,activation='sigmoid')(x_poole)

model = Model(inputs=InceptionV3_base_model.input,outputs=final_predect)
layer_to_Freeze=276    

for layer in model.layers[:layer_to_Freeze]:

    layer.trainable =False

for layer in model.layers[layer_to_Freeze:]:

    layer.trainable=True



sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)





#model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])





model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
nb_train_samples = len(X_train)

nb_validation = len(X_val)

batch_size = 50
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



val_datagen = ImageDataGenerator(    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)
train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)

validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

history_transfer_learning = model.fit_generator(train_generator,epochs=20,

                                                steps_per_epoch=nb_train_samples//batch_size,

                                                validation_data=validation_generator,

                                                validation_steps=nb_validation//batch_size,

                                                verbose=1,

                                                callbacks=my_callback)

fig, ax = plt.subplots(2,1)

ax[0].plot(history_transfer_learning.history['loss'], color='b', label="Training loss")

ax[0].plot(history_transfer_learning.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history_transfer_learning.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history_transfer_learning.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
Y_pred = model.predict(np.array(X_val))

for i in range(0,5):

    if Y_pred[i] >= 0.5: 

        print('I am  sure this is a Dog')

    else: 

        print('I am  sure this is a Cat')

        

    plt.imshow(X_val[i])    

    plt.show()
def prepare_data_test(list_of_images):



    x = [] # images as arrays

    for image in list_of_images:

        x.append(cv2.resize(cv2.imread(image), (150,150), interpolation=cv2.INTER_CUBIC))

        return x
local_zip_test = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

zip_ref = zipfile.ZipFile(local_zip_test, 'r')

zip_ref.extractall('/kaggle/working')

zip_ref.close()

filenames = os.listdir("/kaggle/working/test")

categories = []

for filename in filenames:

    categories.append(filename)

images_list = []

for i in categories:

    i='/kaggle/working/test/'+i

    images_list.append(i)
test_image=prepare_data_test(images_list)
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(np.array(test_image), batch_size=batch_size)


prediction_probabilities = model.predict_generator(test_generator, verbose=1,steps = 12500)
prediction_probabilities
submission_df = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

for i, fname in enumerate(categories):

    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])

    submission_df.at[index-1, 'label'] = prediction_probabilities[i]

submission_df.to_csv('submission.csv', index=False)

submission_df.head()
score = model.evaluate_generator(validation_generator,steps=len(validation_generator))

print('Test score:', score[0])

print('Test accuracy:', score[1])