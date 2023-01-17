from tqdm import tqdm

import cv2

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import random

random.seed(42)
submission=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
train.head()
test.head()
submission.head()
from tensorflow.keras.preprocessing.image import img_to_array

train_img=[]

train_label=[]

path='/kaggle/input/plant-pathology-2020-fgvc7/images'

for im in tqdm(train['image_id']):

    im=im+".jpg"

    final_path=os.path.join(path,im)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(200,200))

    img=img_to_array(img)

    train_img.append(img)
test_img=[]

path='/kaggle/input/plant-pathology-2020-fgvc7/images'

for im in tqdm(test['image_id']):

    im=im+".jpg"

    final_path=os.path.join(path,im)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(200,200))

    img=img_to_array(img)

    test_img.append(img)
train_label=train.loc[:,'healthy':'scab']

from sklearn.model_selection import train_test_split

train_img=np.array(train_img, dtype="float")/255.0

test_img=np.array(test_img, dtype="float")/255.0

train_label=np.array(train_label)

(trainX, testX, trainY, testY) = train_test_split(train_img,

	train_label, test_size=0.2, random_state=42)

print(trainX.shape)

print(testX.shape)

print(trainY.shape)

print(testY.shape)

print(test_img.shape)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False,

        shear_range=0.2,

        fill_mode="nearest")  # randomly flip images

train_datagen.fit(trainX)



valid_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False,

        shear_range=0.2,

        fill_mode="nearest")  # randomly flip images

valid_datagen.fit(testX)

from tensorflow.keras.applications import VGG16, DenseNet201, ResNet152V2, NASNetLarge



from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, BatchNormalization

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.regularizers import l2
base_model=DenseNet201(include_top=False, weights='imagenet',input_shape=(200,200,3))

#base_model.load_weights(r"/kaggle/input/nasnetlargenotop/NASNet-large-no-top.h5")



model=Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(1024,activation='relu', kernel_regularizer=l2(l=0.03)))

model.add(Dropout(0.3))

model.add(Dense(128,activation='relu', kernel_regularizer=l2(l=0.03)))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(Dense(4,activation='softmax'))





from tensorflow.keras.optimizers import Adam



for layer in base_model.layers[:-10]:

    layer.trainable = False

reduce_learning_rate = ReduceLROnPlateau(monitor='val_acc',

                                         factor=0.1,

                                         patience=5,

                                         cooldown=1,

                                         min_lr=0.000001,

                                         verbose=1)

earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, 

                          baseline=None, restore_best_weights=True)



callbacks = [reduce_learning_rate, earlystop]

    





model.compile( optimizer=Adam(lr = 0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

H = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=64), validation_data=valid_datagen.flow(testX, testY, batch_size=32),

                    steps_per_epoch=len(trainX) // 64, epochs=300, callbacks=callbacks)
N = len(H.history['loss'])

plt.plot(np.arange(0, N), H.history['loss'])

plt.plot(np.arange(0, N), H.history['val_loss'])

plt.plot(np.arange(0, N),H.history['accuracy'])

plt.plot(H.history['val_accuracy'])

plt.title('model performance')

plt.ylabel('loss/ acc')

plt.xlabel('epoch')

plt.legend(['loss', 'val-loss', 'acc', 'val-acc'], loc='upper left')

plt.show()
for layer in base_model.layers:

    layer.trainable = True

model.compile( optimizer=Adam(lr = 0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

H1 = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=64), validation_data=valid_datagen.flow(testX, testY, batch_size=32),

                    steps_per_epoch=len(trainX) // 64, epochs=25)
plt.plot(H1.history['loss'])

plt.plot(H1.history['val_loss'])

plt.plot(H1.history['accuracy'])

plt.plot(H1.history['val_accuracy'])

plt.title('model performance')

plt.ylabel('loss/ acc')

plt.xlabel('epoch')

plt.legend(['loss', 'val-loss', 'acc', 'val-acc'], loc='upper left')

plt.show()
y_pred=model.predict(test_img)

print(y_pred)
submission.loc[:,'healthy':'scab']=y_pred
submission.head()
submission.to_csv('submission.csv',index=False)