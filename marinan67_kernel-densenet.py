import pandas as pd

import numpy as np

import keras

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint

from PIL import Image

import os

import glob

import cv2

from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense,Convolution2D,MaxPooling2D,AveragePooling2D,BatchNormalization, SeparableConv2D, GlobalAveragePooling2D

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score

import seaborn as sns



print(os.listdir("../input"))
train = pd.read_csv('../input/labels/labels_new.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print(train.shape, test.shape)

train.head()
cv_img = []

title_font = {'fontname':'Arial', 'size':'14'}

IMG_FOLDER_PREFIX_i = "../input/optos-all/optos_all/optos_all/"

IMG_EXTENSION = ".png"

NUM_IMAGES = 8



for i in range(NUM_IMAGES):

    image_path = IMG_FOLDER_PREFIX_i + str(train['id_code'][i])

    n = cv2.imread(image_path)

    cv_img.append(n)



import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))

for i in range(NUM_IMAGES):

    plt.subplot(4,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.text(0.0, 0.0, 'diagnosis'+str(train['diagnosis'][i]), **title_font)

    plt.imshow(cv_img[i], cmap=plt.cm.binary)
def append_png(fn):

    return fn+".png"

'''train["id_code"]=train["id_code"].apply(append_png)

train.head()'''
new_width  = 224

new_height = 224

        

def preprocess_img(img):

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (3,3))

    img = np.uint8(img)

    for c in range(0, 2):

        img[:,:,c] = cv2.equalizeHist(img[:,:,c])

    

     # convert image to LAB color model

    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)



    # split the image into L, A, and B channels

    l_channel, a_channel, b_channel = cv2.split(image_lab)



    # apply CLAHE to lightness channel

    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))

    cl = clahe.apply(l_channel)



    # merge the CLAHE enhanced L channel with the original A and B channel

    merged_channels = cv2.merge((cl, a_channel, b_channel))



    # convert iamge from LAB color model back to RGB color model

    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR) 

    img = cv2.resize(final_image, (new_width,new_height), interpolation = cv2.INTER_AREA)

    return img  
def process_img(img):

    #img = cv2.cvtColor(img, cv2 . COLOR_BGR2RGB)

    img = cv2.GaussianBlur ( img , ( 5 , 5 ), 0 )

    img = np.uint8(img)

    r, g, b = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

    red = clahe.apply(r)

    gr = clahe.apply(g)

    bl = clahe.apply(b)

    merged_channels = cv2.merge((red, gr, bl))

    merged_channels = cv2.resize(merged_channels , (new_width,new_height), interpolation = cv2.INTER_AREA)

    

    bits_per_channel = 8

    assert merged_channels.dtype == np.uint8



    shift = 8-bits_per_channel

    halfbin = (1 << shift) >> 1



    return ((merged_channels.astype(int) >> shift) << shift) + halfbin    
from keras.preprocessing.image import ImageDataGenerator

valid=0.25

train_datagen = ImageDataGenerator(preprocessing_function= process_img, validation_split=valid,

                                   width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')
import matplotlib.pyplot as plt
cv_img = []

title_font = {'fontname':'Arial', 'size':'14'}

IMG_FOLDER_PREFIX_i = "../input/optos-all/optos_all/optos_all/"

#IMG_EXTENSION = ".png"

NUM_IMAGES = 8



for i in range(NUM_IMAGES):

    image_path = IMG_FOLDER_PREFIX_i + str(train['id_code'][i])

    n = cv2.imread(image_path)

    n = process_img(n)

    cv_img.append(n)

    

plt.figure(figsize=(20, 20))

for i in range(NUM_IMAGES):

    plt.subplot(4,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.text(0.0, 0.0, 'diagnosis'+str(train['diagnosis'][i]), **title_font)

    plt.imshow(cv_img[i], cmap=plt.cm.binary)
import tensorflow as tf

run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = False)
train['diagnosis']=train['diagnosis'].apply(str)
data_train, data_test, labels_train, labels_test = train_test_split(train, train['diagnosis'], 

                                                                    test_size=0.25)

print(len(data_train), len(data_test))
path = '../input/optos-all/optos_all/optos_all/'

train_generator = train_datagen.flow_from_dataframe(dataframe=data_train,  directory=path, 

    x_col='id_code', y_col="diagnosis", target_size=(new_width,new_height), batch_size=50, class_mode='categorical',

                                                    subset='training',seed=42, drop_duplicates=True)



validation_generator = train_datagen.flow_from_dataframe(dataframe=data_train, directory=path,  

  x_col='id_code', y_col="diagnosis",target_size=(new_width,new_height), batch_size=50, class_mode='categorical',

                                                         subset='validation',seed=42, drop_duplicates=True) 
#filepath = '\\kaggle\\working\\'

#mc = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,save_best_only=True, save_weights_only=False, mode='auto', period=1)

early_stopping_callback = EarlyStopping(monitor='val_acc', patience=3, restore_best_weights=True)
from keras.layers import Input

input_tensor = Input(shape=(new_width,new_height,3)) 

base_n=keras.applications.vgg19.VGG19(include_top=False,

     weights='../input/keraspretrainedmodel/keras-pretrain-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',

                                      input_tensor=input_tensor)

model_i = Sequential()

model_i.add(base_n)



model_i.add(Flatten())

model_i.add(Dense(512))

model_i.add(Activation('relu'))

model_i.add(Dropout(0.5))

model_i.add(Dense(5))

model_i.add(Activation('softmax'))
from keras.optimizers import Adam

model_i.compile(loss='categorical_crossentropy',  optimizer=Adam(lr=1e-5), metrics=['accuracy'], options = run_opts)

#model_i.summary()
nb_epoch=25

batch_size_train=50

steps_per_epoch = (len(train)*(1-valid)//batch_size_train)    

print(steps_per_epoch)

validation_steps=len(train)*valid// batch_size_train

print(validation_steps)

model_i.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch, validation_data=validation_generator,

                    validation_steps=validation_steps,epochs=nb_epoch, 

                      callbacks=[early_stopping_callback])



print("Обучение остановлено на эпохе", early_stopping_callback.stopped_epoch)
keras.backend.eval(model_i.optimizer.lr.assign(0.000001))
nb_epoch=25

batch_size_train=32

steps_per_epoch = (len(train)*(1-valid)//batch_size_train)    

print(steps_per_epoch)

validation_steps=len(train)*valid// batch_size_train

print(validation_steps)

model_i.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch, validation_data=validation_generator,

                    validation_steps=validation_steps,epochs=nb_epoch, 

                      callbacks=[early_stopping_callback])



print("Обучение остановлено на эпохе", early_stopping_callback.stopped_epoch)
test_data=ImageDataGenerator(preprocessing_function= process_img)
batch_size = 32

data_test_generator = test_data.flow_from_dataframe(dataframe=data_test,  directory=path, 

    x_col='id_code', y_col="diagnosis", target_size=(new_width,new_height), batch_size= batch_size, class_mode=None, 

                                                    seed=42, shuffle=False)
data_test_generator.reset()

pred_i = model_i.predict_generator (data_test_generator, steps = (len(data_test)//batch_size+1), verbose = 1)
predict=np.argmax(pred_i,axis=1)
labels = labels_test

my_list = labels.values

my_list= [int(item) for item in my_list]

print(my_list[:20])
from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

multiclass = confusion_matrix(my_list, predict)

class_names = ['0', '1', '2', '3', '4']



fig, ax = plot_confusion_matrix(conf_mat=multiclass,

                                colorbar=True,

                                show_absolute=True,

                                show_normed=True,

                                class_names=class_names)

plt.show()
from sklearn.metrics import r2_score

r2_score(my_list, predict)
from sklearn.metrics import classification_report

print(classification_report(my_list, predict))
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test.head()
test["id_code"]=test["id_code"].apply(append_png)

test.head()
test_datagen=ImageDataGenerator(preprocessing_function= process_img)
test_generator=test_datagen.flow_from_dataframe(dataframe=test, 

        directory='../input/aptos2019-blindness-detection/test_images', 

        x_col="id_code", y_col="diagnosis", target_size=(new_width,new_height), batch_size=batch_size, seed=42, 

                                                class_mode=None, shuffle=False)
test_generator.reset()

i_test = model_i.predict_generator (test_generator, steps = (len(test)//batch_size+1), verbose = 1)
i_test=np.argmax(i_test,axis=1)
sample_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_submission.head()
sample_submission["diagnosis"] = i_test

sample_submission.head()
sample_submission.to_csv('submission.csv',index = False)