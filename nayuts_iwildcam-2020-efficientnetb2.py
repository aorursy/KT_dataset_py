!pip install tensorflow-gpu==1.14.0

!pip install keras==2.2.4
import os, glob

import random

from sklearn.model_selection import train_test_split

import cv2

import numpy as np

import pandas as pd

import multiprocessing

from copy import deepcopy

from sklearn.metrics import precision_recall_curve, auc

import keras

import keras.backend as K

from keras.optimizers import Adam

from keras.applications.densenet import DenseNet201

from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, applications

from keras.models import Model, load_model

from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras.utils import Sequence

import matplotlib.pyplot as plt

from IPython.display import Image

from tqdm import tqdm_notebook as tqdm

import json

from numpy.random import seed

seed(10)



%matplotlib inline
!pip install git+https://github.com/qubvel/efficientnet
!ls ../input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json
import json

ann_file = '../input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json'

with open(ann_file) as data_file:

        train_anns = json.load(data_file)
train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]

train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})

df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')

df_train_file_cat['category_id']=df_train_file_cat['category_id'].astype(str)

df_train_file_cat.head()
# Example of images for category_id = 400

img_names = df_train_file_cat[df_train_file_cat['category_id']=='73']['file_name'][:30]



plt.figure(figsize=[15,15])

i = 1

for img_name in img_names:

    img = cv2.imread("../input/iwildcam-2020-fgvc7/train/%s" % img_name)[...,[2, 1, 0]]

    plt.subplot(6, 5, i)

    plt.imshow(img)

    i += 1

plt.show()
#nb_classes = 572

nb_classes = 267

batch_size = 256

img_size = 96

nb_epochs = 10
%%time

train_datagen=ImageDataGenerator(rescale=1./255, 

    validation_split=0.25,

    horizontal_flip = True,    

    zoom_range = 0.3,

    width_shift_range = 0.3,

    height_shift_range=0.3

    )



train_generator=train_datagen.flow_from_dataframe(    

    dataframe=df_train_file_cat,    

    directory="../input/iwildcam-2020-fgvc7/train",

    x_col="file_name",

    y_col="category_id",

    batch_size=batch_size,

    shuffle=True,

    classes = [ str(i) for i in range(nb_classes)],

    class_mode="categorical",    

    target_size=(img_size,img_size))
%%time

test_datagen = ImageDataGenerator(rescale=1./255)



valid_generator=test_datagen.flow_from_dataframe(    

    dataframe=df_train_file_cat[50000:],    

    directory="../input/iwildcam-2020-fgvc7/train",

    x_col="file_name",

    y_col="category_id",

    batch_size=batch_size,

    shuffle=True,

    classes = [ str(i) for i in range(nb_classes)],

    class_mode="categorical",  

    target_size=(img_size,img_size))
import efficientnet.keras as efn 

def get_model():

    K.clear_session()

    base_model =  efn.EfficientNetB2(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))

    x = base_model.output

    predictions = Dense(nb_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=predictions)



model = get_model()


model.compile(optimizers.rmsprop(lr=0.003, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])
# Callbacks



early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
%%time

history = model.fit_generator(generator=train_generator,  

                                    

                                    steps_per_epoch=5,

                                    

                                    validation_data=valid_generator, 

                                    

                                    validation_steps=2,

                                    

                                    epochs=nb_epochs,

                                    callbacks = [early],

                                    verbose=2)
#with open('history.json', 'w') as f:

#    json.dump(history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
import gc

del train_datagen, train_generator

gc.collect()
sam_sub_df = pd.read_csv('../input/iwildcam-2020-fgvc7/sample_submission.csv')
sam_sub_df["file_name"] = sam_sub_df["Id"].map(lambda str : str + ".jpg")
sam_sub_df.head()
%%time



test_generator = test_datagen.flow_from_dataframe(      

    

        dataframe=sam_sub_df,    

    

        directory = "../input/iwildcam-2020-fgvc7/test",    

        x_col="file_name",

        target_size = (img_size,img_size),

        batch_size = 1,

        classes = [ str(i) for i in range(nb_classes)],

        shuffle = False,

        class_mode = None

        )
%%time

test_generator.reset()

predict=model.predict_generator(test_generator, steps = len(test_generator.filenames))
len(predict)
predicted_class_indices=np.argmax(predict,axis=1)
predicted_class_indices
sam_sub_df["Category"] = predicted_class_indices

sam_sub_df = sam_sub_df.loc[:,["Id", "Category"]]

sam_sub_df.to_csv("submission.csv",index=False)