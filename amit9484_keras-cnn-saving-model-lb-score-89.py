import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

from keras.models import Sequential

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Convolution2D,BatchNormalization

import gc

from keras.models import clone_model

from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

import cv2

import numpy as np

from keras.models import model_from_json

from keras.preprocessing.image import ImageDataGenerator



stringpath = '/kaggle/input/bengaliai-cv19'

featurepath = "/kaggle/input/bengaliaicv19feather"



modelpath = "/kaggle/input/my-trained-model"



traincsv =  pd.read_csv(stringpath  + r'/train.csv')

OutputCols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']



BS = 100

HEIGHT = 137

WIDTH = 236

IMG_SIZE = 96



def resize(df, size=IMG_SIZE, need_progress_bar=True):

    resized = {}

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            resized[df.index[i]] = (cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size, size), interpolation = cv2.INTER_AREA)).reshape(-1)

    else:

        for i in range(df.shape[0]):

            resized[df.index[i]] = (cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size, size), interpolation = cv2.INTER_AREA)).reshape(-1)

    resized = pd.DataFrame(resized).T

    del df

    gc.collect()

    return resized



aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,

	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,

	horizontal_flip=True, fill_mode="nearest")





############# Model prepartion #############################

model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(Convolution2D(32,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(Convolution2D(32,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(Convolution2D(32,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(BatchNormalization(axis=-1, momentum=0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(64,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(64,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(64,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(Convolution2D(64,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(BatchNormalization(axis=-1, momentum=0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(128,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(128,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(128,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(Convolution2D(128,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(BatchNormalization(axis=-1, momentum=0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(256,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(256,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(256,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(Convolution2D(256,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(BatchNormalization(axis=-1, momentum=0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(512,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(512,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same',activation = 'relu'))

model.add(Convolution2D(512,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(Convolution2D(512,(3,3),input_shape=(IMG_SIZE, IMG_SIZE,1), padding='same', activation = 'relu'))

model.add(BatchNormalization(axis=-1, momentum=0.2))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(activation="relu", units=1024))

model.add(Dense(activation="relu", units=512))



model_root = clone_model(model)

model_vowel = clone_model(model)

model_consonant = clone_model(model)



model_root.add(Dense(activation = 'softmax', units = 168))

model_vowel.add(Dense(activation = 'softmax', units = 11))

model_consonant.add(Dense( activation = 'softmax', units = 7))







model_root.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

model_vowel.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

model_consonant.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])



print("Done  Model prepartion  ")

############# Model prepartion #############################









############ Creating Input #################################

for i in range(4):

    print(i)

    featherdir = featurepath + r'/train_image_data_{}.feather'.format(i)

    trainData = pd.merge(traincsv,  pd.read_feather(featherdir), how='inner', on=['image_id']).drop(['image_id','grapheme'], axis = 1)

   

    alltrainY = trainData[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]

    trainX = trainData.drop(OutputCols, axis=1)

    

    trainX = resize(trainX)

    trainX_df = trainX.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('uint8')

    

    del trainX,trainData  

    gc.collect()

    

    print("Start model Training ")

    

    trainrootY = pd.get_dummies(alltrainY['grapheme_root']).values 

    trainvowelY = pd.get_dummies(alltrainY['vowel_diacritic']).values

    trainconsonantY = pd.get_dummies(alltrainY['consonant_diacritic']).values

    trainX, ValX,trainrootY,valrootY, trainvowelY, valvowelY, trainconsonantY, valconsonantY = train_test_split(trainX_df, trainrootY, trainvowelY, trainconsonantY, test_size=0.2, random_state=400)

            

    model_root.fit_generator(aug.flow(trainX, trainrootY, batch_size=BS), steps_per_epoch = trainX.shape[0] / BS,  epochs=15, validation_data = (ValX, valrootY))

    model_vowel.fit_generator(aug.flow(trainX, trainvowelY, batch_size=BS),steps_per_epoch = trainX.shape[0] / BS, epochs=12, validation_data = (ValX, valvowelY))

    model_consonant.fit_generator(aug.flow(trainX, trainconsonantY, batch_size=BS), steps_per_epoch = trainX.shape[0] / BS, epochs=12, validation_data = (ValX, valconsonantY))

    del trainX,ValX,trainrootY,trainvowelY,trainconsonantY,valrootY,valvowelY,valconsonantY,trainX_df,alltrainY

    gc.collect()

    print("Done model Training ")

############ Creating Input #################################





############ Saving Model ##################################

model_rootjson = model_root.to_json()

model_voweljson = model_vowel.to_json()

model_consonantjson = model_consonant.to_json()



with open("model_root.json", "w") as json_file:

    json_file.write(model_rootjson)

    

with open("model_vowel.json", "w") as json_file:

    json_file.write(model_voweljson)

    

with open("model_consonant.json", "w") as json_file:

    json_file.write(model_consonantjson)

# serialize weights to HDF5

model_root.save_weights("model_root.h5")

model_vowel.save_weights("model_vowel.h5")

model_consonant.save_weights("model_consonant.h5")

print("Saved model to disk")

############ End Saving Model #################################


