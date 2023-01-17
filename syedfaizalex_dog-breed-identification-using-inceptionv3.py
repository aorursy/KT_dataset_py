import os 

import pandas as pd

import numpy as np

from tqdm import tqdm

import cv2

import pathlib

import shutil

from keras.optimizers import Adam

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Conv2D,MaxPooling2D

from keras.layers import BatchNormalization

from keras.layers import Activation,Dense,Flatten

from keras.layers import Dropout

from keras.models import Sequential,load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint,EarlyStopping
#Dividing photos according to breed in appropriate folders (total 120 breed so 120 folders)

df_train = pd.read_csv('../input/labels.csv')

targets_series = pd.Series(df_train['breed'])

one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)

i = 0 

for f, breed in tqdm(df_train.values):

    pathlib.Path('kaggle.com/syedfaizalex/Dog_Breed_data/{}'.format(breed)).mkdir(parents=True,

                                                                                  exist_ok=True)

    shutil.copy('../input/train/{}.jpg'.format(f),

                'kaggle.com/syedfaizalex/Dog_Breed_data/{}/{}.jpg'.format(breed,f))

    
#Moving 10% photos of each class for validation Dataset

#taking folder names available in folder "Dog_Breed_data" in variable ls

ls = os.listdir('kaggle.com/syedfaizalex/Dog_Breed_data')

for i in tqdm(ls):        

    count = 0

    n =  len(os.listdir('kaggle.com/syedfaizalex/Dog_Breed_data/'+i))

    #taking only 10% photos from each folder/breed for validation

    n = int((n*10)/100)

    for j in os.listdir('kaggle.com/syedfaizalex/Dog_Breed_data/'+i):

        pathlib.Path('kaggle.com/syedfaizalex/Val-Dog-Breed/'+i).mkdir(parents=True, exist_ok=True)

        count+=1 

        if count < n:

            shutil.move('kaggle.com/syedfaizalex/Dog_Breed_data/{}/{}'.format(i,j),

                        'kaggle.com/syedfaizalex/Val-Dog-Breed/{}/{}'.format(i,j))

        else:

            break 
#Using Keras ImageDataGenerator to apply Data augmentation technique in order to increase dataset

print("For Training")

train_x = ImageDataGenerator(rescale=1/255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



train = train_x.flow_from_directory(directory='kaggle.com/syedfaizalex/Dog_Breed_data/',

                                    batch_size=32,target_size=(320,320)) 

print("For Validation")

valid = ImageDataGenerator(rescale=1/255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

valid = valid.flow_from_directory(directory='kaggle.com/syedfaizalex/Val-Dog-Breed',

                                  batch_size=32,target_size=(320,320)) 

#Lets check class indices

print(train.class_indices)
#Tranfer Learning With InceptionV3

model_base = InceptionV3(weights='imagenet',include_top=False,input_shape=(320,320,3))

for i in model_base.layers:

    i.trainable = False

model = Sequential()

model.add(model_base)

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(120))

model.add(Activation('softmax'))

adam=Adam(lr=0.0001)

model.compile(optimizer=adam,loss="categorical_crossentropy",metrics=['accuracy'])

cb1 = ModelCheckpoint(filepath='kaggle.com/syedfaizalex/checkpoint_120_InceptionV3_12-29_3.h5',

                      save_best_only=True)

#Each epoch will take approx. 25Hours if used powerful machines

#after 2-3 epoch you may achieve more than 85% val-acc and more than 95% train acc

history = model.fit_generator(train,validation_data=valid,callbacks=[cb1],

                              steps_per_epoch=9374,epochs=5,validation_steps=848,initial_epoch=0)

history =  pd.DataFrame(history.history).to_csv('kaggle.com/syedfaizalex/trainHistoryDict.csv')

model.save(filepath='kaggle.com/syedfaizalex/CNN_KaggleOnly_Inception_12-31_3.h5')