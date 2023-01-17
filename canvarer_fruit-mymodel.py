import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras 
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import glob
import os
training_fruit_img = []
training_label = []
for dir_path in glob.glob("../input/fruits-360/Training/*"):
    img_label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        training_fruit_img.append((np.array(image) - np.mean(image)) / np.std(image))
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)
label_to_id = {v:k for k,v in enumerate(np.unique(training_label)) }
id_to_label = {v:k for k,v in label_to_id.items() }
id_to_label
training_label_id = np.array([label_to_id[i] for i in training_label])
training_label_id
training_fruit_img.shape,training_label_id.shape
test_fruit_img=[]
test_label =[]
for dir_path in glob.glob("../input/fruits-360/Test/*"):
    img_label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_fruit_img.append((np.array(image) - np.mean(image)) / np.std(image))
        test_label.append(img_label)
test_fruit_img = np.array(test_fruit_img)
test_label = np.array(test_label)
test_label_id = np.array([label_to_id[i] for i in test_label])
test_fruit_img.shape,test_label_id.shape
X_train,X_test = training_fruit_img,test_fruit_img
Y_train,Y_test =training_label_id,test_label_id

X_flat_train = X_train.reshape(X_train.shape[0],64,64,1)
X_flat_test = X_test.reshape(X_test.shape[0],64,64,1)

#One Hot Encode the Output
Y_train = keras.utils.to_categorical(Y_train, 120)
Y_test = keras.utils.to_categorical(Y_test, 120)

print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print('Flattened:', X_flat_train.shape, X_flat_test.shape)
print(X_train[1200].shape)
plt.imshow(X_train[1200])
plt.show()
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dropout, Flatten, Dense,Input, BatchNormalization
from keras.layers import AveragePooling2D,MaxPooling2D,concatenate, Activation, Add
from keras.optimizers import Adam, RMSprop, SGD, Adamax
def residual_conv_block(inp, filters, data_format='channels_last', name=None):

    inp_res = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=(1,1),
        padding='same',
        data_format=data_format,
        name=f'Res_{name}' if name else None)(inp)

    x = BatchNormalization(name=f'BatcgNorm_1_{name}' if name else None)(inp)

    x = Activation('relu', name=f'Relu_2_{name}' if name else None)(x)
    x = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv2D_2_{name}' if name else None)(x)
  
    out = Add(name=f'Out_{name}' if name else None)([x, inp_res])

    #print("Out shape_",out)
    return out
def model_Fruit():
    inp = Input(shape=(64,64,1))

    m = Conv2D(filters=24, kernel_size=(5, 5), activation='relu')(inp)
    m = Dropout(rate=0.4)(m)
    m = BatchNormalization()(m)
    m = AveragePooling2D((2,2))(m)
    m = residual_conv_block(m, 16, name="residual_block_1")
    
    m = Conv2D(filters=12, kernel_size=(3, 3),	activation='relu')(m)
    m = Dropout(rate=0.4)(m)
    m = BatchNormalization()(m)
    m = AveragePooling2D()(m)
    m = residual_conv_block(m, 8, name="residual_block_2")
    
    m = Flatten()(m)
    m = Dense(units=1024, activation='tanh')(m)
    m = Dropout(rate=0.4)(m)
    m = Dense(units=512, activation='tanh')(m)
    out = Dense(units=120, activation = 'softmax')(m)

    model = Model(inp, out)
    model.summary()
    return model

model = model_Fruit()
model.compile(loss='categorical_crossentropy',
             optimizer = Adam(1e-3,beta_1=0.9,beta_2=0.999),
             metrics=['accuracy'])
model.fit(X_flat_train,
          Y_train,
          batch_size=1024,
          epochs=10,
          verbose=1,
         )
score = model.evaluate(X_flat_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
