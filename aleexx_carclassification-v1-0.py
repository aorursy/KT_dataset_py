import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

import pickle

import csv

import os

import keras

import cv2



from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.callbacks import Callback

from keras.regularizers import l2

from keras import optimizers

from keras.models import Model

from keras.utils import np_utils

from keras.applications.xception import Xception

from keras.layers import *



from sklearn.model_selection import train_test_split, StratifiedKFold



import PIL

from PIL import ImageOps, ImageFilter

#увеличим дефолтный размер графиков

from pylab import rcParams

rcParams['figure.figsize'] = 10, 5

#графики в svg выглядят более четкими

%config InlineBackend.figure_format = 'svg' 

%matplotlib inline



print(os.listdir("../input"))
# В сетап выношу основные настройки, так удобней их перебирать в дальнейшем



EPOCHS               = 5

BATCH_SIZE           = 64

LR                   = 1e-4



CLASS_NUM            = 10

IMG_SIZE             = 224

IMG_CHANNELS         = 3

input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)



DATA_PATH = '../input/'

PATH = "../working/car/"
#os.makedirs(PATH,exist_ok=False)



RANDOM_SEED = 431



np.random.seed(RANDOM_SEED)



from tensorflow import set_random_seed

set_random_seed(RANDOM_SEED)
#функция преобразования в чб + подготовка для сетки

def prepareImageToNet(image):

    image = np.array(image)   

    return keras.applications.xception.preprocess_input(

        cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

                    )
# Аугментация данных очень важна когда у нас не большой датасет (как в нашем случае)

# Поиграйся тут параметрами чтоб понять что к чему. 

# Официальная дока https://keras.io/preprocessing/image/



train_datagen = ImageDataGenerator(

    #rescale=1. / 255,

    #preprocessing_function = keras.applications.xception.preprocess_input,

    preprocessing_function = prepareImageToNet,

    rotation_range = 10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range = 0.1,

    shear_range=0.03,

    brightness_range=[0.5, 1.5],

    fill_mode='reflect',

    validation_split=0.1, # set validation split

    horizontal_flip=True)



test_datagen = ImageDataGenerator(

    #rescale=1. / 255

    #preprocessing_function = keras.applications.xception.preprocess_input

    preprocessing_function = prepareImageToNet

)



# Задание для Про - попробуй подключить сторонние более продвинутые библиотеки аугминтации изображений
sample_submission = pd.read_csv(DATA_PATH+"sample_ submission.csv")



# "Заворачиваем" наши данные в generator



train_generator = train_datagen.flow_from_directory(

    DATA_PATH+'train/train/',

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=True, seed=RANDOM_SEED,

    subset='training') # set as training data



test_generator = train_datagen.flow_from_directory(

    DATA_PATH+'train/train/',

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=True, seed=RANDOM_SEED,

    subset='validation') # set as validation data



test_sub_generator = test_datagen.flow_from_dataframe(

    dataframe=sample_submission,

    directory=DATA_PATH+'test/test_upload',

    x_col="Id",

    y_col=None,

    shuffle=False,

    class_mode=None,

    seed=RANDOM_SEED,

    target_size=(IMG_SIZE, IMG_SIZE),

    batch_size=BATCH_SIZE,)



# кстати, ты заметил, что для сабмишена мы используем другой источник для генератора flow_from_dataframe? 

# Как ты думаешь, почему?
import tensorflow as tf



# чистит сессию в Keras и TF

def reset_tf_session():

    curr_session = tf.get_default_session()

    # close current session

    if curr_session is not None:

        curr_session.close()

    # reset graph

    K.clear_session()

    # create new session

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    s = tf.InteractiveSession(config=config)

    K.set_session(s)

    return s
# Рекомендую добавть еще функции из https://keras.io/callbacks/

checkpoint = ModelCheckpoint('best_model.hdf5', monitor = ['val_acc'], verbose = 1, mode = 'max', save_best_only = False)

callbacks_list = [checkpoint]



# Для про - попробуй добавить разные техники управления Learning Rate

# Например:

# https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6

# http://teleported.in/posts/cyclic-learning-rate/
import keras



def makeTrainable(model, N, learn_rate):

    # все слои обучаемы

    for layer in model.layers:

        layer.trainable = True

        if isinstance(layer, keras.layers.BatchNormalization):

            # быстрее настраиваем параметры батч норма!

            layer.momentum = 0.9



    if N != None:

        # fine-tuning только для N последних слоев

        for layer in model.layers[:-N]:

            # батч норм должен настраивать свои параметры для новых данных! а иначе фиксируем слой!

            if not isinstance(layer, keras.layers.BatchNormalization):

                layer.trainable = False        



    #

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learn_rate), metrics=["accuracy"])    
def fit(model, Epochs=EPOCHS):

    # Обучаем

    history = model.fit_generator(

            train_generator,

            steps_per_epoch = len(train_generator),

            validation_data = test_generator, 

            validation_steps = len(test_generator),

            epochs = Epochs,

            callbacks = callbacks_list

    )

    
s = reset_tf_session()
base_model = Xception(weights='imagenet', include_top=False, input_shape = input_shape)
# Устанавливаем новую "голову"

# Тут тоже можно поиграться, попробуй добавить Batch Normalization например.



x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(256, activation='elu')(x)

x = BatchNormalization()(x)

x = Dropout(0.25)(x)



# and a logistic layer -- let's say we have 10 classes

predictions = Dense(CLASS_NUM, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)
makeTrainable(model, None, 1e-4)

fit(model)
#model.save('../working/model_step1.hdf5')
#model.load_weights('../working/model_step1.hdf5')



makeTrainable(model, 20, 3e-3)

fit(model, 10)
#model.save('../working/model_step2.hdf5')
makeTrainable(model, 50, 1e-4)

fit(model, 20)
makeTrainable(model, 150, 1e-4)

fit(model, 20)
#model.save('../working/model_step3.hdf5')
scores = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
test_sub_generator.reset()

predictions = model.predict_generator(test_sub_generator, steps=len(test_sub_generator), verbose=1) 

predictions = np.argmax(predictions, axis=-1) #multiple categories

label_map = (train_generator.class_indices)

label_map = dict((v,k) for k,v in label_map.items()) #flip k,v

predictions = [label_map[k] for k in predictions]
filenames_with_dir=test_sub_generator.filenames

submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])

submission['Id'] = submission['Id'].replace('test_upload/','')

submission.to_csv('submission.csv', index=False)

print('Save submit')



# Для Про - попробуй TTA