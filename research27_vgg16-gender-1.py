import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import utils

import os







# Networks

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16



from keras.preprocessing.image import ImageDataGenerator



# Layers

from keras.layers import Dense, Activation, Flatten, Dropout

from keras import backend as K



# Other

from keras import optimizers

from keras import losses

from keras.optimizers import SGD, Adam

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.models import load_model





# Utils

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random, glob

import os, sys, csv

import cv2

import time, datetime



print(os.listdir("../input/gender-data-1/gender_data_1/gender_data_1/"))

BATCH_SIZE = 32

WIDTH = 299

HEIGHT = 299

FC_LAYERS = [1024, 1024]

TRAIN_DIR = "../input/gender-data-1/gender_data_1/gender_data_1/train"

VAL_DIR = "../input/gender-data-1/gender_data_1/gender_data_1/val"

TEST_DIR = "../input/gender-data-1/gender_data_1/gender_data_1/test"
preprocessing_function = None

base_model = None

from keras.applications.vgg16 import preprocess_input

preprocessing_function = preprocess_input

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
train_datagen =  ImageDataGenerator(

      preprocessing_function=preprocessing_function

)

val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)



train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)



validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)
class_list = utils.get_subfolders(TRAIN_DIR)

utils.save_class_list(class_list, model_name="VGG16", dataset_name="dataset")



finetune_model = utils.build_finetune_model(base_model, dropout=0.0001, fc_layers=FC_LAYERS, num_classes=len(class_list))



adam = Adam(lr=0.00001)

finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

num_train_images = utils.get_num_files(TRAIN_DIR)

num_val_images = utils.get_num_files(VAL_DIR)



def lr_decay(epoch):

    if epoch%20 == 0 and epoch!=0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr/2)

        print("LR changed to {}".format(lr/2))

    return K.get_value(model.optimizer.lr)



learning_rate_schedule = LearningRateScheduler(lr_decay)



filepath="VGG16" + "_model_weight_gender_ep_6.h5"

checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')

callbacks_list = [checkpoint]



history = finetune_model.fit_generator(train_generator, epochs=6, workers=8, steps_per_epoch=num_train_images // BATCH_SIZE, 

validation_data=validation_generator, validation_steps=num_val_images // BATCH_SIZE, class_weight='auto', shuffle=True, callbacks=callbacks_list)



utils.plot_training(history)
from os import listdir

from os.path import isfile, join

import time, datetime



i = -1

cls_name_arr = ['male','female']

cm = [[0.0]*2] * 2



print(cm)

class_list_file = "VGG16" + "_" + "dataset" + "_class_list.txt"

class_list = utils.load_class_list(class_list_file)

finetune_model = utils.build_finetune_model(base_model, dropout=0.0001, fc_layers=FC_LAYERS, num_classes=len(class_list))



finetune_model.load_weights("VGG16" + "_model_weight_gender_ep_6.h5")



for cls_name in cls_name_arr:

    i+=1

    onlyfiles = [f for f in listdir(TEST_DIR+'/' + cls_name + '/') if isfile(join(TEST_DIR+'/' + cls_name + '/', f))]

    cnt_male = 0.0

    cnt_female = 0.0



    

    for img in onlyfiles:

        



        image = cv2.imread(join(TEST_DIR+'/'+ cls_name + '/',img),cv2.IMREAD_COLOR)

        save_image = image

        image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))

        image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))



        st = time.time()



        out = finetune_model.predict(image)



        confidence = out[0]

        class_prediction = list(out[0]).index(max(out[0]))

        class_name = class_list[class_prediction]



        run_time = time.time()-st

        #print(img)



        print("Predicted class = ", class_name[0])

        #print("Confidence = ", confidence)

        if(class_name[0] == 'male'):

            cnt_male += 1

        elif(class_name[0] == 'female'):

            cnt_female += 1

        

            #print("Run time = ", run_time)

            #cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)

        #cv2.imwrite("Predictions/" + class_name[0] + ".png", save_image)

    cnt_male = (cnt_male/955.0)*100.0

    cnt_female = (cnt_female/955.0)*100.0

    

    print("total: ", len(onlyfiles))

    #print("predicted: ", cnt)

    cm[i] = [cnt_male,cnt_female]

    print(cm)

print(cm)



import seaborn as sn

import pandas as pd

import matplotlib.pyplot as plt



       

df_cm = pd.DataFrame(cm, cls_name_arr,

                  cls_name_arr)

plt.figure(figsize = (12,12))

sn.set(font_scale=1.4)#for label size

sn.heatmap(df_cm,cmap="Blues", annot=True,annot_kws={"size": 12})# font size

plt.savefig('VGG17_gender.png')

plt.show()