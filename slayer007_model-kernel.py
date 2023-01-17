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
# start=time.time()

# train0=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

# print(f"TRAIN0 SHAPE: {train0.shape}  and it took {time.time()-start} seconds")
# current=time.time()

# train1=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

# print(f"TRAIN1 SHAPE: {train1.shape}  and it took {time.time()-current} seconds")

# current=time.time()

# train2=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

# print(f"TRAIN2 SHAPE: {train2.shape}  and it took {time.time()-current} seconds")

# current=time.time()

# train3=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')

# print(f"TRAIN3 SHAPE: {train3.shape}  and it took {time.time()-current} seconds")
# from pathlib import Path

# featherdir = Path('/kaggle/input/bengaliaicv19feather')

# start=time.time()

# train0 = pd.read_feather(featherdir/'train_image_data_0.feather')

# train1 = pd.read_feather(featherdir/'train_image_data_1.feather')

# train2 = pd.read_feather(featherdir/'train_image_data_2.feather')

# train3 = pd.read_feather(featherdir/'train_image_data_3.feather')

# print(f"Time to load {time.time()-start} seconds")
# del train0

# del train1

# del train2

# del train3

# import gc

# gc.collect()
# gc.collect()
print("PARQUET READ")
# train0.to_feather('train0.feather')

# del train0

# train1.to_feather('train1.feather')

# del train1

# train2.to_feather('train2.feather')

# del train2

# train3.to_feather('train3.feather')

# del train3



# train0=train0.iloc[:,1:]

# train1=train1.iloc[:,1:]

# train2=train2.iloc[:,1:]

# train3=train3.iloc[:,1:]
from tqdm import tqdm

import cv2

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
# def reduce_mem_usage(df):

#     """ 

#     iterate through all the columns of a dataframe and 

#     modify the data type to reduce memory usage.        

#     """

#     start_mem = df.memory_usage().sum() / 1024**2

#     print(('Memory usage of dataframe is {:.2f}' 

#                      'MB').format(start_mem))

    

#     for col in df.columns:

#         col_type = df[col].dtype

        

#         if col_type != object:

#             c_min = df[col].min()

#             c_max = df[col].max()

#             if str(col_type)[:3] == 'int':

#                 if c_min > np.iinfo(np.int8).min and c_max <\

#                   np.iinfo(np.int8).max:

#                     df[col] = df[col].astype(np.int8)

#                 elif c_min > np.iinfo(np.int16).min and c_max <\

#                    np.iinfo(np.int16).max:

#                     df[col] = df[col].astype(np.int16)

#                 elif c_min > np.iinfo(np.int32).min and c_max <\

#                    np.iinfo(np.int32).max:

#                     df[col] = df[col].astype(np.int32)

#                 elif c_min > np.iinfo(np.int64).min and c_max <\

#                    np.iinfo(np.int64).max:

#                     df[col] = df[col].astype(np.int64)  

#             else:

#                 if c_min > np.finfo(np.float16).min and c_max <\

#                    np.finfo(np.float16).max:

#                     df[col] = df[col].astype(np.float16)

#                 elif c_min > np.finfo(np.float32).min and c_max <\

#                    np.finfo(np.float32).max:

#                     df[col] = df[col].astype(np.float32)

#                 else:

#                     df[col] = df[col].astype(np.float64)

#         else:

#             df[col] = df[col].astype('category')

#     end_mem = df.memory_usage().sum() / 1024**2

#     print(('Memory usage after optimization is: {:.2f}' 

#                               'MB').format(end_mem))

#     print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 

#                                              / start_mem))

    

#     return df
#### train0

featherdir = Path('/kaggle/input/bengaliaicv19feather')

start=time.time()

train0=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

# train0=pd.read_feather(featherdir/'train_image_data_0.feather')

print(f"TRAIN0 SHAPE: {train0.shape}  and it took {time.time()-start} seconds")

train0=train0.iloc[:,1:]

train0_r=resize(train0)

train0_r=(train0_r-np.mean(train0_r))/(train0_r.max()-train0_r.min())

import gc

del train0

gc.collect()
train0_r.shape
train0_r.head()
# import gc

# del train0

# gc.collect()
###### train1

current=time.time()

train1=pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

# train1=pd.read_feather(featherdir/'train_image_data_1.feather')

print(f"TRAIN1 SHAPE: {train1.shape}  and it took {time.time()-current} seconds")

train1=train1.iloc[:,1:]

# train1_r=resize(train1)/255

train1_r=resize(train1)

train1_r=(train1_r-np.mean(train1_r))/(train1_r.max()-train1_r.min())

del train1

gc.collect()

train1_r.head()
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

# train2_r=resize(train2)/255

train2_r=resize(train2)

train2_r=(train2_r-np.mean(train2_r))/(train2_r.max()-train2_r.min())

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

# train3_r=resize(train3)/255

train3_r=resize(train3)

train3_r=(train3_r-np.mean(train3_r))/(train3_r.max()-train3_r.min())

del train3

gc.collect()


# del train3

# gc.collect()
print("resized train")
xtrain0=train0_r.to_numpy()

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

xtrain=xtrain.reshape(-1,64,64,1).astype('float32')

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

model.add(layers.Dense(1000))

model.add(layers.LeakyReLU(alpha=0.1))



model.add(layers.BatchNormalization())

model.add(layers.Dense(512))

model.add(layers.LeakyReLU(alpha=0.1))



model.add(layers.BatchNormalization())



# # MODEL

# model = Sequential()

# model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(64, 64, 1)))

# model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu'))

# model.add(BatchNormalization(momentum=0.15))

# model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu'))

# model.add(BatchNormalization(momentum=0.15))

# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu'))

# model.add(Dropout(rate=0.3))



# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

# model.add(BatchNormalization(momentum=0.15))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

# model.add(BatchNormalization(momentum=0.15))

# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu'))

# model.add(Dropout(rate=0.3))



# model.add(Flatten())

# model.add(Dense(1000, activation = "relu"))

# model.add(Dropout(0.20))

# model.add(Dense(512, activation = "relu"))

# model.add(Dropout(0.20))
from keras.models import clone_model
model_root=clone_model(model)

model_consonant=clone_model(model)

model_vowel=clone_model(model)
model_root.add(Dense(168, activation = 'softmax'))

model_vowel.add(Dense(11, activation = 'softmax'))

model_consonant.add(Dense(7, activation = 'softmax'))
model_root.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

model_vowel.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

model_consonant.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
model_root.summary()
# learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 

#                                             patience=3, 

#                                             verbose=1,

#                                             factor=0.5, 

#                                             min_lr=0.00001)
# datagen = ImageDataGenerator(

#             featurewise_center=False,  # set input mean to 0 over the dataset

#             samplewise_center=False,  # set each sample mean to 0

#             featurewise_std_normalization=False,  # divide inputs by std of the dataset

#             samplewise_std_normalization=False,  # divide each input by its std

#             zca_whitening=False,  # apply ZCA whitening

#             rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

#             zoom_range = 0.15, # Randomly zoom image 

#             width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

#             height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

#             horizontal_flip=False,  # randomly flip images

#             vertical_flip=False)  # randomly flip images





#         # This will just calculate parameters required to augment the given data. This won't perform any augmentations

# datagen.fit(xtrain)
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

    rotation_range=12, 

    zoom_range=0.2, 

    width_shift_range=0.2, 

    height_shift_range=0.2,

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

model_root.fit_generator(datagen.flow(X_train, Y_train, batch_size=128),

                              steps_per_epoch=len(X_train)//128,

                              epochs=40,

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
# y_consonant = to_categorical(y_consonant, 7)

y_consonant.shape
y_consonant = pd.get_dummies(y_consonant)

y_consonant=y_consonant.to_numpy()
X_train, X_val, Y_train, Y_val = shuffle(xtrain, y_consonant, 5)
# X_train, X_val, Y_train, Y_val = train_test_split(xtrain,y_consonant, test_size=0.08, random_state=42)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=12, 

    zoom_range=0.2, 

    width_shift_range=0.2, 

    height_shift_range=0.2,

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

model_consonant.fit_generator(datagen.flow(X_train, Y_train, batch_size=128),

                              steps_per_epoch=len(X_train)//128,

                              epochs=40,

                              validation_data=(np.array(X_val),np.array(Y_val)),

                              validation_steps=50,

                              callbacks=[learning_rate_reduction, es])

# model_consonant.fit(X_train, Y_train, batch_size=1024, epochs=20, validation_data=(X_val, Y_val))
del X_train

gc.collect()

del X_val

gc.collect()

del Y_train

gc.collect()

del Y_val

gc.collect()
del y_consonant

gc.collect()
print("consonant done !")
########################## VOWEL ######################
# y_vowel = to_categorical(y_vowel, 11)

y_vowel.shape
y_vowel = pd.get_dummies(y_vowel)
y_vowel=y_vowel.to_numpy()
X_train, X_val, Y_train, Y_val = shuffle(xtrain, y_vowel, 5)
# X_train, X_val, Y_train, Y_val = train_test_split(xtrain,y_vowel, test_size=0.08, random_state=42)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=12, 

    zoom_range=0.2, 

    width_shift_range=0.2, 

    height_shift_range=0.2,

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

model_vowel.fit_generator(datagen.flow(X_train, Y_train, batch_size=128),

                              steps_per_epoch=len(X_train)//128,

                              epochs=40,

                              validation_data=(np.array(X_val),np.array(Y_val)),

                              validation_steps=50,

                              callbacks=[learning_rate_reduction, es])

# model_vowel.fit(X_train, Y_train, batch_size=1024, epochs=20, validation_data=(X_val, Y_val))
del X_train

del X_val

del Y_train

del Y_val

gc.collect()
del y_vowel

gc.collect()
print("vowel done !")
# model_dict = {

#     'grapheme_root': model_root,

#     'vowel_diacritic': model_vowel,

#     'consonant_diacritic': model_consonant

# }
del xtrain

del train

gc.collect()
import pickle
Pkl_root = "model_root.pkl" 

Pkl_consonant = "model_consonant.pkl"  

Pkl_vowel = "model_vowel.pkl"  





with open(Pkl_root, 'wb') as file:  

    pickle.dump(model_root, file)

with open(Pkl_consonant, 'wb') as file:  

    pickle.dump(model_consonant, file)

with open(Pkl_vowel, 'wb') as file:  

    pickle.dump(model_vowel, file)    

    
# preds_dict = {

#     'grapheme_root': [],

#     'vowel_diacritic': [],

#     'consonant_diacritic': []

# }
# components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

# target=[] # model predictions placeholder

# row_id=[] # row_id place holder

# for i in tqdm(range(4)):

#     df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

#     df_test_img.set_index('image_id', inplace=True)



#     X_test = resize(df_test_img, need_progress_bar=True)/255

#     X_test = X_test.values.reshape(-1, 64, 64, 1)



#     for pred in preds_dict:

#         preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)



#     for k,id in enumerate(df_test_img.index.values):  

#         for i,comp in enumerate(components):

#             id_sample=id+'_'+comp

#             row_id.append(id_sample)

#             target.append(preds_dict[comp][k])

#     del df_test_img

#     del X_test

#     gc.collect()



# df_sample = pd.DataFrame(

#     {

#         'row_id': row_id,

#         'target':target

#     },

#     columns = ['row_id','target'] 

# )

# df_sample.to_csv('submission.csv',index=False)

# # del target

# # del row_id

# # gc.collect()

# df_sample.head()
print("Model DONE !")