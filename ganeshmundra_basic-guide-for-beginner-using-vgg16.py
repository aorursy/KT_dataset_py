from tqdm import tqdm

import cv2

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt
submission=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
train.head()
test.head()
submission.head()
train_img=[]

train_label=[]

path='/kaggle/input/plant-pathology-2020-fgvc7/images'

for im in tqdm(train['image_id']):

    im=im+".jpg"

    final_path=os.path.join(path,im)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(224,224))

    img=img.astype('float32')

    train_img.append(img)
test_img=[]

path='/kaggle/input/plant-pathology-2020-fgvc7/images'

for im in tqdm(test['image_id']):

    im=im+".jpg"

    final_path=os.path.join(path,im)

    img=cv2.imread(final_path)

    img=cv2.resize(img,(224,224))

    img=img.astype(('float32'))

    test_img.append(img)
train_label=train.loc[:,'healthy':'scab']

train_img=np.array(train_img)

test_img=np.array(test_img)

train_label=np.array(train_label)
print(train_img.shape)

print(test_img.shape)

print(train_label.shape)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(train_img)

from tensorflow.keras.applications.vgg16 import VGG16



from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization



from tensorflow.keras.callbacks import ReduceLROnPlateau
base_model=VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3), pooling='avg')



model=Sequential()

model.add(base_model)



model.add(Dense(256,activation='relu'))

model.add(Dense(4,activation='softmax'))





from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop



base_model.trainable=False



reduce_learning_rate = ReduceLROnPlateau(monitor='loss',

                                         factor=0.1,

                                         patience=2,

                                         cooldown=2,

                                         min_lr=0.00001,

                                         verbose=1)



callbacks = [reduce_learning_rate]

    





model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(datagen.flow(train_img, train_label, batch_size=32),

                    epochs=50,callbacks=callbacks)
y_pred=model.predict(test_img)

print(y_pred)
submission.loc[:,'healthy':'scab']=y_pred
submission.head()
submission.to_csv('submission.csv',index=False)