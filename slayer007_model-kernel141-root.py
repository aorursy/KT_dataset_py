# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from pathlib import Path# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test=pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

sample_sub=pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
# from pathlib import Path

# featherdir = Path('/kaggle/input/bengaliaicv19feather')

# start=time.time()

# train0 = pd.read_feather(featherdir/'train_image_data_0.feather')

# train1 = pd.read_feather(featherdir/'train_image_data_1.feather')

# train2 = pd.read_feather(featherdir/'train_image_data_2.feather')

# train3 = pd.read_feather(featherdir/'train_image_data_3.feather')

# print(f"Time to load {time.time()-start} seconds")
from tqdm import tqdm

import cv2

# IMG_SIZE=128

# def resize(df, size=IMG_SIZE, need_progress_bar=True):

#     resized = {}

#     if need_progress_bar:

#         for i in tqdm(range(df.shape[0])):

#             img0 = 255 - df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH).astype(np.uint8)

#             #normalize each image by its max val

#             img = (img0*(255.0/img0.max())).astype(np.uint8)

#             img = crop_resize(img0, size=size)

#             img = ((img.astype(np.float32)/255.0) - 0.0692)/0.2051

#             resized[df.index[i]] = img.reshape(-1)

# #             resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

#     else:

#         for i in range(df.shape[0]):

#             img0 = 255 - df.loc[df.index[i]].values.reshape(HEIGHT, WIDTH).astype(np.uint8)

#             #normalize each image by its max val

# #             img = (img0[i]).astype(np.uint8)

#             img = (img0*(255.0/img0.max())).astype(np.uint8)

#             img = crop_resize(img0, size=size)

#             img = ((img.astype(np.float32)/255.0) - 0.0692)/0.2051

#             resized[df.index[i]] = img.reshape(-1)

#     return pd.DataFrame(resized).T

def resize(df, size=64, need_progress_bar=True):

    resized = {}

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

            

    else:

        for i in range(df.shape[0]):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

            

    resized = pd.DataFrame(resized).T

    resized.columns=resized.columns.astype(str)

    return resized
#### train0

# x_tot,x2_tot = [],[]



# featherdir = Path('/kaggle/input/bengaliaicv19feather')

start=time.time()

train0=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

# train0=pd.read_feather(featherdir/'train_image_data_0.feather')

print(f"TRAIN0 SHAPE: {train0.shape}  and it took {time.time()-start} seconds")

train0=train0.iloc[:,1:]

train0_r=resize(train0)/255



# data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

# for idx in tqdm(range(len(df))):

#     name = df.iloc[idx,0]

#     #normalize each image by its max val

#     img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)

#     img = crop_resize(img)



#     x_tot.append((img/255.0))

#     x2_tot.append(((img/255.0)**2)) 

# df = pd.read_parquet(fname)

        #the input is inverted



import gc

del train0

gc.collect()
###### train1

current=time.time()

train1=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

# train1=pd.read_feather(featherdir/'train_image_data_1.feather')

print(f"TRAIN1 SHAPE: {train1.shape}  and it took {time.time()-current} seconds")

train1=train1.iloc[:,1:]

train1_r=resize(train1)/255

# train1_r=resize(train1)

# train1_r=(train1_r-np.mean(train1_r))/(train1_r.max()-train1_r.min())

del train1

gc.collect()

train0_r.head()
################################ CODE TO CHECK MEMORY USAGE################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# gl.info(memory_usage='deep')


# del train1
# gc.collect()
######## train2

current=time.time()

train2=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

# train2=pd.read_feather(featherdir/'train_image_data_2.feather')

print(f"TRAIN2 SHAPE: {train2.shape}  and it took {time.time()-current} seconds")

train2=train2.iloc[:,1:]

train2_r=resize(train2)/255

# train2_r=resize(train2)

# train2_r=(train2_r-np.mean(train2_r))/(train2_r.max()-train2_r.min())

del train2

gc.collect()



# del train2

# gc.collect()
########### train3

current=time.time()

train3=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

# train3=pd.read_feather(featherdir/'train_image_data_3.feather')

print(f"TRAIN3 SHAPE: {train3.shape}  and it took {time.time()-current} seconds")

train3=train3.iloc[:,1:]

train3_r=resize(train3)/255

# train3_r=resize(train3)

# train3_r=(train3_r-np.mean(train3_r))/(train3_r.max()-train3_r.min())

del train3

gc.collect()


# del train3

# gc.collect()
print("resized train")
xtrain0=(train0_r).to_numpy()

del train0_r

gc.collect()

xtrain1=train1_r.to_numpy()

del train1_r

gc.collect()
xtrain2=train2_r.to_numpy()

del train2_r

gc.collect()
xtrain3=train3_r.to_numpy()

del train3_r

gc.collect()
xtrain=np.concatenate((xtrain0,xtrain1,xtrain2,xtrain3),axis=0)

del xtrain1

del xtrain2

del xtrain3

del xtrain0

gc.collect()
xtrain.shape
y_root=train['grapheme_root'].values

y_vowel=train['vowel_diacritic'].values

y_consonant=train['consonant_diacritic'].values

xtrain=xtrain.reshape(-1,64,64,1).astype(np.float32)

y_vowel.shape
############################# IMPORTING LIBRARIES CNN ##########################

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation,BatchNormalization,Flatten,Conv2D,MaxPool2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras import layers,regularizers

from keras.optimizers import Adam,RMSprop

model = Sequential()

model.add(layers.Conv2D(64, (3,3), padding='same', input_shape=(64, 64, 1)))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.Conv2D(64,  (3,3), padding='same'))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.Conv2D(64,  (3,3), padding='same'))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))



model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Dropout(0.2))



model.add(layers.Conv2D(128, (3,3), padding='same'))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.Conv2D(128, (3,3), padding='same'))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.Conv2D(128, (3,3), padding='same'))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))



model.add(layers.MaxPooling2D(2,2))

model.add(layers.Dropout(0.2)) 



model.add(layers.Conv2D(256, (3,3), padding='same'))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.Conv2D(256, (3,3), padding='same'))

model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(layers.LeakyReLU(alpha=0.1))



model.add(layers.MaxPooling2D(2,2))

model.add(layers.Dropout(0.2))





model.add(layers.Flatten())

model.add(layers.Dense(1024))

model.add(layers.LeakyReLU(alpha=0.1))



model.add(layers.BatchNormalization())

model.add(layers.Dense(256))

model.add(layers.LeakyReLU(alpha=0.1))



model.add(layers.BatchNormalization())



from keras.models import clone_model
model_root=clone_model(model)

model_consonant=clone_model(model)

model_vowel=clone_model(model)
model_root.add(Dense(168, activation = 'softmax'))

model_vowel.add(Dense(11, activation = 'softmax'))

model_consonant.add(Dense(7, activation = 'softmax'))
# from keras.optimizers import Adam,RMSprop

# optimizer = RMSprop(learning_rate=0.002,rho=0.9)
model_root.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])

model_vowel.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])

model_consonant.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])
model_root.summary()
print("model created")
######################### ROOT #########################
from tensorflow.keras.utils import to_categorical
y_root.shape
y_root = pd.get_dummies(y_root)

# xx = to_categorical(y_root, 168)
y_root=y_root.to_numpy()
type(y_root)
200840/3
def shuffle(matrix, target, test_proportion):

    ratio = int(matrix.shape[0]/test_proportion) #should be int

    X_train = matrix[ratio:,:]

    X_test =  matrix[:ratio,:]

    Y_train = target[ratio:,:]

    Y_test =  target[:ratio,:]

    return X_train, X_test, Y_train, Y_test



X_train, X_val, Y_train, Y_val = shuffle(xtrain, y_root, 5)
print('reached')


# from sklearn.model_selection import train_test_split

# X_train, X_val, Y_train, Y_val = train_test_split(xtrain,y_root, test_size=0.08, random_state=42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=8, 

    zoom_range=0.15, 

    width_shift_range=0.15, 

    height_shift_range=0.15,

#     rescale=1./255

)

datagen.fit(X_train)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau( 

    monitor='loss',    # Quantity to be monitored.

    factor=0.5,       # Factor by which the learning rate will be reduced. new_lr = lr * factor

    patience=3,        # The number of epochs with no improvement after which learning rate will be reduced.

    verbose=1,         # 0: quiet - 1: update messages.

    mode="auto",       # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 

                       # in the max mode it will be reduced when the quantity monitored has stopped increasing; 

                       # in auto mode, the direction is automatically inferred from the name of the monitored quantity.

    min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes.

    cooldown=0,        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.

    min_lr=0     # lower bound on the learning rate.

    )



es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=300, restore_best_weights=True)

model_root.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),

                              steps_per_epoch=len(X_train)//32,

                              epochs=50,

                              validation_data=(np.array(X_val),np.array(Y_val)),

                              validation_steps=50,

                              callbacks=[learning_rate_reduction, es])

# model_root.fit(X_train, Y_train, batch_size=1024, epochs=20, validation_data=(X_val, Y_val))
del X_train

del X_val

del Y_train

del Y_val

gc.collect()
del y_root

gc.collect()
print("root done !")
##################### CONSONANT ##################################
del xtrain

del train

gc.collect()
import pickle
Pkl_root = "model_root.pkl" 

# Pkl_consonant = "model_consonant.pkl"  

# Pkl_vowel = "model_vowel.pkl"  





with open(Pkl_root, 'wb') as file:  

    pickle.dump(model_root, file)

# with open(Pkl_consonant, 'wb') as file:  

#     pickle.dump(model_consonant, file)

# with open(Pkl_vowel, 'wb') as file:  

#     pickle.dump(model_vowel, file)    

    
print("Model DONE !")