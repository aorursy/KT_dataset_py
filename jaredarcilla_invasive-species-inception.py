import pandas as pd

import numpy as np

from PIL import Image, ImageFile

import time

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint
train_path = '../input/train/'

train_labels = pd.read_csv('../input/train_labels.csv')

print(train_labels.head())
ImageFile.LOAD_TRUNCATED_IMAGES=True

def get_image(path):

    img = Image.open(path)

    img = img.resize((289,217))

    return img
x_train = np.zeros((1377,217,289,3))

x_val = np.zeros((459,217,289,3))

y_train = np.array(train_labels.invasive.values[0:1377]) 

y_val = np.array(train_labels.invasive.values[1377:1836])
for i in range(1377):

    file_no = i+1

    x_train[i] = get_image(train_path + str(file_no) + '.jpg')

    if i % 100 == 0:

        print('train pics finished: ' + str(i))

for i in range(459):

    file_no = i+1378

    x_val[i] = get_image(train_path + str(file_no) + '.jpg')

    

    if i % 100 == 0:

        print('validation pics finished: ' + str(i))

train_datagen = ImageDataGenerator(

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        rescale=1./255,

        zoom_range=0.1,

        horizontal_flip=True,

        fill_mode='nearest')



valid_datagen = ImageDataGenerator(

        rescale=1./255)
#Kaggle does not support downloading weights

base_model = applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(217,289,3))
pred = base_model.output

pred = GlobalAveragePooling2D()(pred)

pred = Dense(1024, activation='relu')(pred)

pred = Dropout(0.4)(pred)

pred = Dense(1,activation='softmax')(pred)

model=Model(inputs=base_model.input, outputs=pred)
for layer in base_model.layers:

    layer.trainable = False
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_datagen.flow(x_train,y_train, batch_size=32),

                    steps_per_epoch = len(x_train)//32, epochs = 100,

                    validation_data = valid_datagen.flow(x_val,y_val, batch_size=32),

                    validation_steps = len(y_val)//32,

                    verbose=1)