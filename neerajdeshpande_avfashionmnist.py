import os

import numpy as np
import pandas as pd
import cv2
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, ZeroPadding2D, Conv2D, GlobalMaxPooling2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Activation, AveragePooling2D, Dropout, SeparableConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

X_train = np.load("../input/fashionmnist/X_train.npy", allow_pickle=True)
X_test = np.load("../input/fashionmnist/X_test.npy", allow_pickle=True)
y_train = np.load("../input/fashionmnist/y_train.npy", allow_pickle=True)
X_train = np.expand_dims(X_train, axis=-1)  # Equivalent to x[:,:,np.newaxis]
X_test = np.expand_dims(X_test, axis=-1)
X_test = X_test/255
y_train = y_train[:,-1]
X_train.shape
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train = train_datagen.flow(X_train, y_train, batch_size=32)
X_train.shape

earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
y_train.shape
num_labels = len(set(y_train))
epochs = 20
batch_size = 32
WIDTH = HEIGHT = 28
CHANNELS = 1
model1 = Sequential(name='6Conv')
model1.add(Input(shape=(WIDTH, HEIGHT, CHANNELS)))
model1.add(Conv2D(64, kernel_size=(3,3)))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Conv2D(64, kernel_size=(3,3)))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Conv2D(128, kernel_size=(3, 3)))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Conv2D(256, kernel_size=(3, 3)))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(1024, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(512, activation='relu'))
model1.add(Dense(num_labels, activation='softmax'))

mcp_save = tf.keras.callbacks.ModelCheckpoint('6Conv.h5', save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False)

adam = Adam(lr=0.0001, decay=1e-6)
model1.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model1.fit(train, epochs=40, steps_per_epoch=60000//32)


model1.save('model1.h5')
#8Layer FCNN
model2 = Sequential(name='DenseNet')
model2.add(Input(shape=(WIDTH, HEIGHT, CHANNELS)))
model2.add(Flatten())
model2.add(Dense(1024, activation='relu'))
model2.add(Dense(1024, activation='relu'))
model2.add(Dense(512, activation='relu'))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(num_labels, activation='softmax'))

mcp_save = tf.keras.callbacks.ModelCheckpoint('DenseNet.h5', save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False)

adam = Adam()
model2.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model2.fit(train, epochs=200, steps_per_epoch=40000//32, callbacks=[mcp_save, reduce_lr_loss], validation_data = ((X_train2/255)[:1000],y_train2[:1000]))
#0.8882
model2.save('model2.h5')
# model = Sequential(name='Fashionista')
# model.add(Input(shape=(WIDTH, HEIGHT, CHANNELS)))
# model.add(Conv2D(64, kernel_size=(7,7), strides=(1,1), activation='relu'))
# model.add(Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
# model.add(Flatten())
# model.add(Dropout(0.3))
# model.add(Dense(1024,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(num_labels, activation='softmax'))
# #0.9157
# model = ResNet50(input_shape = (28, 28, 1), classes = 10)

# Mini Resnet - 0.92

inputs = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(32, 3)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64, 3)(x)
x = layers.BatchNormalization()(x)
block_1_output = layers.Activation('relu')(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(block_1_output)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

block_1_output =layers.Conv2D(128, 3, activation='relu', padding='same')(block_1_output)
block_1_output = layers.BatchNormalization()(block_1_output)
block_1_output = layers.Activation('relu')(block_1_output)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(256, 3, padding="same")(block_2_output)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(256, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

block_2_output =layers.Conv2D(256, 3, activation='relu', padding='same')(block_2_output)
block_2_output = layers.BatchNormalization()(block_2_output)
block_2_output = layers.Activation('relu')(block_2_output)
block_3_output = layers.add([x, block_2_output])
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation="relu")(x)
outputs = layers.Dense(10, activation='softmax')(x)

resnet = keras.Model(inputs, outputs, name="resnet")

mcp_save = tf.keras.callbacks.ModelCheckpoint('ResNet.h5', save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False)

adam = Adam(lr=0.0001)
resnet.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

resnet.summary()
# resnet.fit(train, epochs=60, steps_per_epoch=60000//32)

resnet.save('resnet.h5')
# adam = Adam(lr=0.0001, decay=1e-6)
# #91.35

# import numpy as np
# from keras import layers
# from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# from keras.models import Model, load_model
# from keras.preprocessing import image
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
# from keras.initializers import glorot_uniform
# import scipy.misc
# from matplotlib.pyplot import imshow

# import keras.backend as K
# K.set_image_data_format('channels_last')
# K.set_learning_phase(1)



# def identity_block(X, f, filters, stage, block):
#     """

#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network

#     Returns:
#     X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
#     """

#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'

#     # Retrieve Filters
#     F1, F2, F3 = filters

#     # Save the input value. You'll need this later to add back to the main path.
#     X_shortcut = X

#     # First component of main path
#     X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
#     X = Activation('relu')(X)


#     # Second component of main path (≈3 lines)
#     X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
#     X = Activation('relu')(X)

#     # Third component of main path (≈2 lines)
#     X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)


#     return X

# def convolutional_block(X, f, filters, stage, block, s = 2):
#     """

#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#     s -- Integer, specifying the stride to be used

#     Returns:
#     X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
#     """

#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'

#     # Retrieve Filters
#     F1, F2, F3 = filters

#     # Save the input value
#     X_shortcut = X


#     ##### MAIN PATH #####
#     # First component of main path
#     X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
#     X = Activation('relu')(X)

#     # Second component of main path (≈3 lines)
#     X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
#     X = Activation('relu')(X)


#     # Third component of main path (≈2 lines)
#     X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


#     ##### SHORTCUT PATH #### (≈2 lines)
#     X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
#                         kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
#     X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)


#     return X


# def ResNet50(input_shape=(64, 64, 3), classes=6):
#     """
#     Implementation of the popular ResNet50 the following architecture:
#     CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
#     -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

#     Arguments:
#     input_shape -- shape of the images of the dataset
#     classes -- integer, number of classes

#     Returns:
#     model -- a Model() instance in Keras
#     """

#     # Define the input as a tensor with shape input_shape
#     X_input = Input(input_shape)

#     # Zero-Padding
#     X = ZeroPadding2D((3, 3))(X_input)

#     # Stage 1
#     X = Conv2D(64, (7, 7), strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((3, 3), strides=(2, 2))(X)

#     # Stage 2
#     X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

#     ### START CODE HERE ###

#     # Stage 3 (≈4 lines)
#     X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

#     # Stage 4 (≈6 lines)
#     X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

#     # Stage 5 (≈3 lines)
#     X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
#     X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
#     X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

#     # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"

#     ### END CODE HERE ###

#     # output layer
#     X = Flatten()(X)
#     X = Dense(1024, activation = 'relu')(X)
#     X = Dropout(0.2)(X)
#     X = Dense(512, activation = 'relu')(X)
#     X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)


#         # Create model
#     model = Model(inputs = X_input, outputs = X, name='ResNet50')

#     return model

# model = ResNet50(input_shape = (28, 28, 1), classes = 10)

# model.compile(optimizer=adam,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()
# adam = Adam(lr=0.01, decay=0.001)

# model.compile(optimizer=adam,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# resnet.compile(optimizer=adam,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model = keras.models.load_model('../input/fashionmnistmodels/FashionistaResnet50.h5')
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/resnet/")
# model.summary()
# |model.fit_generator(it, steps_per_epoch=60000/32, val)
# model.fit(train, epochs=100, steps_per_epoch=59000//32, validation_steps=1000//32, validation_data=val)
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.get_steps_per_epoch(),
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     epochs=100)
model_a = load_model('./6Conv.h5')
model_b = load_model('./DenseNet.h5')
model_c = load_model('./ResNet.h5')

loss, acc = model1.evaluate(X_train2/255, y_train2)
print(f'Loss1 = {loss}, Accuracy1 = {acc}')
loss, acc = model2.evaluate(X_train2/255, y_train2)
print(f'Loss2 = {loss}, Accuracy2 = {acc}')
loss, acc = resnet.evaluate(X_train2/255, y_train2)
print(f'Loss3 = {loss}, Accuracy3 = {acc}')
loss, acc = model_a.evaluate(X_train2/255, y_train2)
print(f'LossA = {loss}, AccuracyA = {acc}')
loss, acc = model_b.evaluate(X_train2/255, y_train2)
print(f'LossB = {loss}, AccuracyB = {acc}')
loss, acc = model_c.evaluate(X_train2/255, y_train2)
print(f'LossC = {loss}, AccuracyC = {acc}')
# model.save('FashionistDenseNet.h5')
# preds = np.argmax(model.predict(X_test), axis=-1)

# predictions = np.array(model.predict_classes(X_test))
# testdf = pd.read_csv('../input/testquestions/test.csv')
# predictions = np.vstack((testdf.id.values, preds)).T
# predictions[:,1]
# i='resnet'
# pd.DataFrame(data={'id':predictions[:,0], 'label':predictions[:,1]}).set_index('id').to_csv('y_test'+str(i)+'.csv')

# %load_ext tensorboard
# %tensorboard --logdir logs

NUM_MODELS = 3
train_predictions1 = model_a.predict(X_train2/255)
train_predictions2 = model_b.predict(X_train2/255)
train_predictions3 = model_c.predict(X_train2/255)

X_pred_train = np.hstack((train_predictions1, train_predictions2, train_predictions3))
X_pred_train.shape
level1_input = Input(shape=(num_labels*NUM_MODELS))
X = Dense(64, activation='relu')(level1_input)
X = Dense(64, activation='relu')(X)
output = Dense(num_labels, activation='softmax')(X)
level1 = keras.Model(level1_input, output, name="ensembler")

adam = Adam(lr=0.00001, decay=1e-6)
level1.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
level1.summary()
level1.fit(X_pred_train,y_train2, epochs=50, batch_size=32)
ensemble = LogisticRegression(max_iter=1000)
ensemble.fit(X_pred_train,y_train2)
final_preds = ensemble.predict(X_pred_train)

accuracy = accuracy_score(y_train2, final_preds)
accuracy
preds1 = model_a.predict(X_test)
preds2 = model_b.predict(X_test)
preds3 = model_c.predict(X_test)

X_pred_test = np.hstack((preds1, preds2, preds3))
X_pred_test.shape
final_predictions1 = np.argmax(level1.predict(X_pred_test), axis=-1)
final_predictions2 = ensemble.predict(X_pred_test)
testdf = pd.read_csv('../input/testquestions/test.csv')
predictions_df1 = np.vstack((testdf.id.values, final_predictions1)).T
predictions_df1[:,1]
predictions_df2 = np.vstack((testdf.id.values, np.argmax(model1.predict(X_test), axis=-1))).T
predictions_df2
# pd.DataFrame(data={'id':predictions_df1[:,0], 'label':predictions_df1[:,1]}).set_index('id').to_csv('y_test_ensemble_dense.csv')
pd.DataFrame(data={'id':predictions_df2[:,0], 'label':predictions_df2[:,1]}).set_index('id').to_csv('y_test_ensemble_log.csv')

