# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
X_train_ori = np.load('/kaggle/input/rcmemulators/X_train.dat', allow_pickle=True)
temp_mpl_gcm = X_train_ori[:,4,6,15]
Y_train_ori = pd.read_csv('/kaggle/input/rcmemulators/Y_train_mpl.csv')
Ytrain_temp=np.asarray(Y_train_ori.tempé - temp_mpl_gcm)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_ori, Ytrain_temp, test_size = 0.2, random_state = 42)
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_valid = sc_x.transform(X_valid)
X_train = X_train.reshape(X_train.shape[0], 11, 11, 19)
X_valid = X_valid.reshape(X_valid.shape[0], 11, 11, 19)
X_test=np.load('/kaggle/input/rcmemulators/X_test.dat', allow_pickle=True)

import keras
import keras.models as km
import keras.layers as kl
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest'
#     )
def identity_block(inputs, filters):
  layers = Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
  layers = keras.layers.BatchNormalization()(layers)
  layers = Conv2D(filters, (3, 3), padding='same', activation='relu')(layers)
  layers = keras.layers.BatchNormalization()(layers)
  #Convolution with a 1*1 kernel to match dimension 
  skip_co = Conv2D(filters, kernel_size = (1, 1), padding='same', activation='relu')(inputs)
  layers = keras.layers.add([layers, skip_co])
  return layers

def init_CNN(inputs, filters):
  layers = (Conv2D(filters, kernel_size = (1, 1), padding='same', activation='relu', input_shape=(11, 11, 19)))(inputs)
  for k in range(5):
    layers = identity_block(layers, filters)
  layers = MaxPooling2D(pool_size=(2, 2))(layers)
  for k in range(5):
    layers = identity_block(layers, filters)
  layers = MaxPooling2D(pool_size=(2, 2))(layers)
  for k in range(5):
    layers = identity_block(layers, filters)
  layers = MaxPooling2D(pool_size=(2, 2))(layers)

  layers = Flatten()(layers)
  layers = Dense(16, activation='relu')(layers)
  layers = Dense(1, activation = 'linear')(layers)
  return layers

inputs = keras.layers.Input(shape=(11, 11, 19))
outputs = init_CNN(inputs, filters=16)
model = keras.Model(inputs, outputs)
model.compile(loss='mse', optimizer='adadelta')
model.summary()
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# from tensorflow.keras.optimizers import Adam
# model = Sequential()
# model.add(Conv2D(filters = 8, kernel_size = (3,3), padding = 'same', input_shape = (11,11,19), activation = 'relu'))
# model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'same', activation = 'relu'))
# model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'same', activation = 'relu'))
# model.add(Conv2D(filters = 32, kernel_size = (3,3), strides=(2, 2), padding = 'same', activation = 'relu'))
# model.add(Flatten())
# model.add(Dense(16, activation = 'relu'))
# # model.add(Dense(8, activation = 'relu'))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer = Adam(lr=1e-3), metrics=['mae'])
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

epochs = 100
batch_size = 32

callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1),
           EarlyStopping(monitor='val_loss', patience=15, verbose=1)]
#            ModelCheckpoint('CNN.h5', monitor='val_loss', verbose=1, save_best_only=True)]
10
# train_flow = train_datagen.flow(X_train[:1000], y_train[:1000], batch_size=batch_size)
# history = model.fit_generator(train_flow, validation_data=(X_valid[:100], y_valid[:100]), callbacks=callbacks, epochs=epochs)

model.fit(x=X_train[:1000], y=y_train[:1000],
          validation_data=(X_valid[:100], y_valid[:100]),
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks)
import matplotlib.pyplot as plt

# Plot history: MAE
plt.figure(figsize=(12, 6))
plt.title("Loss", fontsize=18)
plt.plot(model.history.history['loss'], 'b',  label='Train loss')
plt.plot(model.history.history['val_loss'], 'r', label='Validation loss')

plt.ylabel('MSE', fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylim((0,3.0))

plt.legend(fontsize=12)
plt.show()
X_test_origin = np.load('/kaggle/input/rcmemulators/X_test.dat', allow_pickle=True)
temp_mpl_gcm=X_test_origin[:,4,6,15]
X_test = X_test_origin.reshape(X_test_origin.shape[0], -1)
X_test = sc_x.transform(X_test)
X_test = X_test.reshape(X_test.shape[0], 11, 11, 19)
previ = model.predict(X_test)
pred = previ[:,0]+ X_test_origin[:,4,6,15] ## we add again 
pred.shape
plt.plot(pred)
res = pd.read_csv('/kaggle/input/rcmemulators/samplesub.csv')

res.tempé = pred

res.to_csv('submission.csv', index=False)
# Y_train_2d = np.load('/kaggle/input/rcmemulators/Y_train_box.dat', allow_pickle=True)
# Y_train_2d.shape
# plt.imshow(Y_train_2d[0,:,:])
# plt.colorbar()
# plt.show()
# temp_2d_gcm=X_train[:,4:6,4:7,15]
# plt.imshow(temp_2d_gcm[0,:,:])
# plt.colorbar()
# mean_box_gcm = temp_2d_gcm.mean(axis=(1,2),keepdims=True)
# mean_box_gcm.shape
# import keras
# from keras.optimizers import Adam
# from keras import layers
# from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Cropping2D

# def block_conv(conv, filters, BN):
#     conv = layers.Conv2D(filters, 3, padding='same')(conv)
#     if BN:
#         conv = layers.BatchNormalization()(conv)
#     conv = layers.Activation('relu')(conv)
#     conv = layers.Conv2D(filters, 3, padding='same')(conv)
#     if BN:
#         conv = layers.BatchNormalization()(conv)
#     conv = layers.Activation('relu')(conv)
#     return conv

# def block_up(conv, filters, BN):
#     conv = block_conv(conv, filters, BN)
#     conv = layers.UpSampling2D(size=(2, 2))(conv)
#     return conv

# def UNET(inputs, BN=False):
#     pool0 = layers.UpSampling2D(size=(3, 3))(inputs)
#     crop0 = layers.Cropping2D(
#     cropping=((0, 1), (0, 1)), input_shape=(33, 33, 19))(pool0)
    
#     conv1 = block_conv(crop0, 64, BN)
#     pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

#     conv2 = block_conv(pool1, 128, BN)
#     pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
#     conv3 = block_up(pool2, 128, BN)
    
#     print(conv3.shape, conv2.shape)
    
#     up3  = layers.concatenate([conv3, conv2])
#     conv4 = block_up(up3, 128, BN)

#     print(conv3.shape, conv4.shape)
    
#     up4  = layers.concatenate([conv4, conv1])
#     conv4 = block_conv(up4, 64, BN)
#     conv5 = layers.Conv2D(1, 1, padding='same', activation='linear')(conv4)

#     return (keras.models.Model(inputs=inputs, outputs=conv5))

# inputs = keras.Input(shape = (11, 11, 19))
# model2 = UNET(inputs, BN=True)
# model2.compile(optimizer=Adam(lr=0.001), loss='mse')
# model2.summary()
# Y_train_2d_diff = Y_train_2d - mean_box_gcm ## We can try to predict the difference the region average in the GCM output
# print(Y_train_2d_diff.shape)
# Y_train2_2d , Y_val_2d = Y_train_2d_diff[ech_train,:,:,None],Y_train_2d_diff[ech_val,:,:,None] 
# plt.imshow(Y_train_2d_diff[0,:,:])
# plt.colorbar()
# def generator_3D(inputs, outputs, batch_size):
#     while 1:
#         dim = 256
#         liste_inputs = np.zeros((batch_size, dim, dim, 2))
#         liste_outputs = np.zeros((batch_size, dim, dim, 1))

#         for nb_field in range(batch_size):

#             numero = np.random.randint(inputs.shape[0])
 

#             abs_x = np.random.randint(0, 498-dim+1)
#             abs_y = np.random.randint(0, 498-dim+1)
        
#             liste_inputs[nb_field, :, :]=inputs[numero, abs_x:abs_x+dim, abs_y:abs_y+dim]
#             #liste_temp[nb_field, :, :, 1]=inputs[abs_x:abs_x+dim, abs_y:abs_y+dim

#             liste_outputs[nb_field, :, :]=outputs[numero, abs_x:abs_x+dim, abs_y:abs_y+dim]
    

#         yield liste_inputs, liste_outputs
# Y_train2_2d.shape, Y_val_2d.shape
# epochs = 100
# batch_size = 32

# train_generator = generator_3D(X_train2, Y_train2_2d, batch_size)
# valid_generator = generator_3D(X_val, Y_val_2d, batch_size)

# callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, verbose=1),
#            EarlyStopping(monitor='val_loss', patience=15, verbose=1)]
# #            ModelCheckpoint('save_model/U-net.h5', monitor='val_loss', verbose=1, save_best_only=True)]

# history2 = model2.fit(X_train2,Y_train2_2d, batch_size=batch_size, validation_data=(X_val, Y_val_2d), epochs=epochs, callbacks=callbacks)
# # history2 = model.fit_generator(train_generator, epochs=20, steps_per_epoch=X_train2.shape[0]/batch_size, validation_data=valid_generator, validation_steps = 1, callbacks=callbacks)
# # Plot history: MAE
# plt.figure(figsize=(12, 6))
# plt.title("Loss", fontsize=18)
# plt.plot(history2.history['loss'], 'b',  label='Train loss')
# plt.plot(history2.history['val_loss'], 'r', label='Validation loss')

# plt.ylabel('MSE', fontsize=15)
# plt.xlabel('Epochs', fontsize=15)

# plt.legend(fontsize=12)
# plt.show()
# temp_2d_gcm_test=X_test[:,4:6,4:7,15]
# mean_box_gcm_test = temp_2d_gcm_test.mean(axis=(1,2),keepdims=True)

# pred_2d = model2.predict(Xtest) + mean_box_gcm_test[:,:,:,None]

# predictions = pred_2d[:,:,:,0]
# predictions.shape
# plt.imshow(pred_2d[10,:,:,0],vmin=260)
# plt.colorbar()
# plt.show()
# def create_submission(predictions):
#     assert predictions.shape==(10220, 32, 32), f"Wrong shape for your prediction file : "\
#                                       f"{predictions.shape} instead of (10220, 32, 32)" 
    
#     if os.path.exists("submission.zip"):
#         !rm submission.zip
#     np.save("y_test", predictions)
#     !mv y_test.npy y_test.predict
#     !zip -r submission.zip y_test_predict.zip
#     print ("Bundle submision.zip created !")
#     return None

# #
# predictions = np.asarray(pred_2d[:,:,:,0])
# create_submission(predictions) ## the output y_test.predict is created then you need to download it and zip it before submit it on codalab
