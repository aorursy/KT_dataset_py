import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
train = pd.read_json("../input/stanford-covid-vaccine/train.json",lines=True)

test = pd.read_json("../input/stanford-covid-vaccine/test.json",lines=True)

ss = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")



train = train.set_index('index')

test = test.set_index('index')
# read npy data file

bpps_dir = '../input/stanford-covid-vaccine/bpps/'

bpps_list = os.listdir(bpps_dir)

bpps_npy = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_list[25]}')

print('Count of npy files: ', len(bpps_list))

print('Size of image: ', bpps_npy.shape)
targets = ['reactivity','deg_Mg_pH10','deg_Mg_50C','deg_pH10','deg_50C']
train = train[['id']+targets ]
train['reactivity'] = train['reactivity'].apply(lambda x: np.mean(x))

train['deg_Mg_pH10'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))

train['deg_Mg_50C'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))

train['deg_pH10'] = train['deg_pH10'].apply(lambda x: np.mean(x))

train['deg_50C'] = train['deg_50C'].apply(lambda x: np.mean(x))
train
train_data_ids = train['id'].values
train_img = []

for ID in train_data_ids:

    img_path = os.path.join(bpps_dir,ID+'.npy')

    img = np.load(img_path)

    train_img.append(img)

y = train[targets].values
train_img = np.array(train_img).reshape(-1, 107, 107, 1)
y.shape
train_img.shape
input_shape = (107, 107, 1)
from sklearn.preprocessing import StandardScaler

# load data

# create scaler

scaler = StandardScaler()

# fit scaler on data

# scaler.fit(data)

from sklearn.preprocessing import MinMaxScaler

# create scaler

scalerM = MinMaxScaler()

# fit scaler on data

# scaler.fit(data)

# apply transform

# standardized = scaler.transform(data)
X_train, X_val, y_train, y_val = train_test_split(train_img, y, test_size=0.2, random_state=32)
X_train.max()
y_train.max()
# X_train = scalerM.transform(X_train)

# y_train = scalerM.transform(y_train)
import tensorflow as tf

import tensorflow.keras.layers as L



from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D, Convolution2D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.utils import plot_model

import tensorflow as tf
from tensorflow.keras import backend as K

def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# Initialising the CNN

model = Sequential()



# 1 - Convolution

model.add(Conv2D(32,(2,2), padding='same', input_shape=input_shape))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



# model.add(ZeroPadding2D((1,1)))

# model.add(MaxPooling2D((2,2), strides=(2,2)))



# model.add(MaxPooling2D((2,2), strides=(2,2)))





# Flattening

model.add(Flatten())



# Fully connected layer

model.add(Dense(16))

model.add(BatchNormalization())

model.add(Activation('elu'))

model.add(Dropout(0.5))

model.add(Dense(5, activation='elu'))



opt = Adam(lr=0.005)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])

model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,

                              patience=2, min_lr=0.00001, mode='auto')

callbacks = [reduce_lr]

history = model.fit(



    x=X_train,

    y=y_train,

    epochs=40,

    validation_data = (X_val,y_val),

    callbacks=callbacks

)
plt.figure(figsize=(15,7))

ax1 = plt.subplot(1,2,1)

ax1.plot(history.history['loss'], color='b', label='Training Loss') 

ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)

legend = ax1.legend(loc='best', shadow=True)

ax2 = plt.subplot(1,2,2)

ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy') 

ax2.plot(history.history['val_accuracy'], color='r', label = 'Validation Accuracy')

legend = ax2.legend(loc='best', shadow=True)
test

test_public = test[test.seq_length == 107]

test_private = test[test.seq_length == 130]

test_public_ids = test_public['id'].values

test_private_ids = test_private['id'].values

test_public_img = []

for ID in test_public_ids:

    img_path = os.path.join(bpps_dir,ID+'.npy')

    img = np.load(img_path)

    test_public_img.append(img)



test_private_img = []

for ID in test_private_ids:

    img_path = os.path.join(bpps_dir,ID+'.npy')

    img = np.load(img_path)

    test_private_img.append(img)

test_public_img = np.array(test_public_img).reshape(-1, 107, 107, 1)

test_private_img = np.array(test_private_img).reshape(-1, 130, 130, 1)

pred_public = model.predict(test_public_img)
len(test_private_img)

pred_public = np.repeat(pred_public,repeats=107,axis=0)

pred_private = np.repeat(np.array([0,0,0,0,0]),repeats=130*3005,axis=0).reshape(-1,5)

prediction = np.concatenate((pred_public,pred_private),axis=0)

sub = pd.DataFrame(prediction)
seqpos = ss.id_seqpos.values

sub['id_seqpos'] = seqpos

sub = sub.rename(columns={0: "reactivity", 1: "deg_Mg_pH10",2: "deg_Mg_50C", 3: "deg_pH10", 4: "deg_50C"})
sub.to_csv("submission.csv",index=False)

# from tensorflow.keras.applications import inception_v3, resnet50

# model = resnet50.ResNet50(weights='imagenet', include_top=False)
# for layer in model.layers:

#     layer.trainable = False
# x = model.output

# x = L.GlobalAveragePooling2D()(x)

# x = L.Dense(128, activation='relu')(x) 

# predictions = L.Dense(5, activation='linear')(x)

# model = Model(model.input, predictions)
# model.compile(

#         optimizer='adam',

#         loss = 'mean_squared_error',

#         metrics=['accuracy'],

#     )


# model.summary()