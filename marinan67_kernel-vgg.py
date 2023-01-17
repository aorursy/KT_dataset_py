import pandas as pd

import numpy as np

from keras.optimizers import SGD

from keras.optimizers import rmsprop

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

from PIL import Image

import os

import glob

import cv2

from keras.models import Sequential

import keras

from keras.layers import Activation, Dropout, Flatten, Dense,Conv2D,Conv3D,MaxPooling2D,AveragePooling2D,BatchNormalization

import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score

import seaborn as sns

import tensorflow as tf



print(os.listdir("../input"))
train_dir = "../input/aerial-cactus-identification/train/train/"

test_dir = "../input/aerial-cactus-identification/test/test/"

train = pd.read_csv('../input/aerial-cactus-identification/train.csv')

train.head()
cv_img = []

for img in glob.glob("../input/aerial-cactus-identification/train/train/*.jpg"):

    n= cv2.imread(img)

    cv_img.append(n)

im = np.asarray(cv_img)    

print(im.shape)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

plt.figure(figsize=(6,6))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.text(0.0, 0.0, 'has_cactus'+str(train['has_cactus'][i]))

    plt.imshow(cv_img[i], cmap=plt.cm.binary)
base_v=keras.applications.vgg16.VGG16(include_top=False,

                weights='../input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',input_shape=(32,32,3))
model_v = Sequential()

model_v.add(base_v)



model_v.add(Flatten())

model_v.add(Dense(256))

model_v.add(Activation('relu'))

model_v.add(Dropout(0.5))

model_v.add(Dense(1))

model_v.add(Activation('sigmoid'))
from keras.optimizers import Adam

model_v.compile(loss='binary_crossentropy',  optimizer=Adam(lr=1e-5), metrics=['accuracy'])

model_v.summary()
from keras.preprocessing.image import ImageDataGenerator

valid=0.2

train_datagen = ImageDataGenerator( rescale=1. / 255, shear_range=0.2, zoom_range=0.2, validation_split=valid)
train['has_cactus']=['1' if x ==1 else '0' for x in train['has_cactus']]



train_generator = train_datagen.flow_from_dataframe(dataframe=train, directory="../input/aerial-cactus-identification/train/train", x_col="id", 

                                            y_col="has_cactus", target_size=(32, 32), batch_size=10, class_mode='binary',

subset='training',seed=42) # set as training data

#same train directory

validation_generator = train_datagen.flow_from_dataframe(dataframe=train, directory="../input/aerial-cactus-identification/train/train", x_col="id", 

                                           y_col="has_cactus", target_size=(32, 32),batch_size=10,class_mode='binary',

subset='validation',seed=42) 
nb_epoch=20

batch_size=40

steps_per_epoch=len(train)*(1-valid)// batch_size

print(steps_per_epoch)

validation_steps=len(train)*valid// batch_size

print(validation_steps)

early_stopping_callback = EarlyStopping(monitor='val_acc', patience=2, restore_best_weights=True)

model_v.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch, validation_data=validation_generator,

                    validation_steps=validation_steps,epochs=nb_epoch, callbacks=[early_stopping_callback])



print("Обучение остановлено на эпохе", early_stopping_callback.stopped_epoch)
keras.backend.eval(model_v.optimizer.lr.assign(0.00001))
nb_epoch=20

batch_size=40

steps_per_epoch=len(train)*(1-valid)// batch_size

print(steps_per_epoch)

validation_steps=len(train)*valid// batch_size

print(validation_steps)

early_stopping_callback = EarlyStopping(monitor='val_acc', patience=2, restore_best_weights=True)

model_v.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch, validation_data=validation_generator,

                    validation_steps=validation_steps,epochs=nb_epoch, callbacks=[early_stopping_callback])



print("Обучение остановлено на эпохе", early_stopping_callback.stopped_epoch)
scores = model_v.evaluate_generator(validation_generator,(len(train)*valid),pickle_safe = False) #len(train)*valid testing images

print("Accuracy = ", scores[1])
from keras.layers import Input

input_tensor = Input(shape=(32,32, 3)) 

base_i=keras.applications.resnet50.ResNet50(include_top=False,

                weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',input_tensor=input_tensor)

model = Sequential()

model.add(base_i)



model.add(Flatten())

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))
from keras.optimizers import Adam

model.compile(loss='binary_crossentropy',  optimizer=Adam(lr=1e-5), metrics=['accuracy'])

model.summary()
nb_epoch=20

batch_size=40

steps_per_epoch=len(train)*(1-valid)// batch_size

print(steps_per_epoch)

validation_steps=len(train)*valid// batch_size

print(validation_steps)

early_stopping_callback = EarlyStopping(monitor='val_acc', patience=2, restore_best_weights=True)

model.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch, validation_data=validation_generator,

                    validation_steps=validation_steps,epochs=nb_epoch, callbacks=[early_stopping_callback])



print("Обучение остановлено на эпохе", early_stopping_callback.stopped_epoch)
testdf = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(dataframe=testdf,x_col = "id", y_col = 'has_cactus',

directory="../input/aerial-cactus-identification/test/test", batch_size=batch_size, seed=42, 

                                                class_mode=None, shuffle=False,target_size=(32,32))
testdf.head()
test_generator.reset () 

pred_v = model_v.predict_generator (test_generator, steps = len(testdf)// batch_size, verbose = 1)



pred = model.predict_generator (test_generator, steps = len(testdf)// batch_size, verbose = 1)
pred_class=[1 if x >= 0.5 else 0 for x in pred]

pred_v_class=[1 if x >= 0.5 else 0 for x in pred_v]

print(pred_class[:25])

print(pred_v_class[:25])
testdf['has_cactus'] = pred_v_class

results_v=testdf

results_v.to_csv("results_v.csv",index=False)

results_v.head(10)

results_r=testdf

results_r['has_cactus'] = pred_class

results_r.to_csv("results_r.csv",index=False)

results_r.head(10)
cv_img = []

for img in glob.glob("../input/aerial-cactus-identification/test/test/*.jpg"):

    n= cv2.imread(img)

    cv_img.append(n)



plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.text(0.0, 0.0, 'has_cactus_v'+str(results_v['has_cactus'][i]))

    plt.imshow(cv_img[i], cmap=plt.cm.binary)