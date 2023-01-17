# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os
print(os.listdir("../input"))
from tqdm import tqdm

# Any results you write to the current directory are saved as output.
# EDA

df =  pd.read_csv("../input/train_labels.csv")
df.info()
each_label = df.groupby("label").count()
each_label = each_label.rename(columns = {"id" : "count"})
each_label = each_label.sort_values("count", ascending=False)
each_label
pos_df = df[df.label==1]
neg_df = df[df.label==0]

sample_size = 65000
random_images_pos = random.sample(pos_df.id.tolist(), sample_size)
random_images_neg = random.sample(neg_df.id.tolist(), sample_size)
id_ls = []
id_ls.extend(random_images_neg)
id_ls.extend(random_images_pos)
len(id_ls)
new_df = pd.DataFrame({"id":id_ls})
new_df = pd.merge(new_df, df, how='inner', on=['id'])
df = new_df.sample(frac=1)

df.info()
import cv2
def load_img(i,path):
    im = cv2.imread(path+i+".tif")
    im = cv2.resize(im,(71,71))
    return im/255
    
# xception
import keras
from sklearn.model_selection import train_test_split
from keras.applications import Xception
from keras.layers import Dense,Flatten
from keras import Sequential
from keras.models import Model
from keras.optimizers import adam
# Parameter
num_class = 2
im_size = 256

base_model = Xception(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))


# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we wi`l train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(adam(lr=0.00001),loss='categorical_crossentropy', 
              metrics=["accuracy"])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen=ImageDataGenerator()
train_df,test_df = train_test_split( new_df, test_size=0.15, random_state=11)
train_gen = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory = "../input/train/",
    x_col = "id",
    y_col = "label",
    has_ext = False,
    shuffle= True)

test_gen = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = "../input/train/",
    x_col = "id",
    y_col = "label",
    has_ext = False,
    shuffle= False)
model.fit_generator(train_gen,steps_per_epoch=30,
                    validation_data=test_gen,
                    validation_steps=30,
                    epochs=25, 
                    verbose=1)
model.save("cancerDetectionXception.h5")
