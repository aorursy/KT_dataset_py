!pip install -q efficientnet
import os

import pandas as pd

import plotly.express as px

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import PIL

from PIL import Image, ImageDraw

import glob

import cv2

import random

from keras.preprocessing.image import ImageDataGenerator

from keras import layers

from keras import Sequential

from keras import models

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import *

from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.applications import VGG19

import efficientnet.tfkeras as efn

from keras.callbacks import ModelCheckpoint

from tensorflow import keras

from keras.models import load_model

import seaborn as sns

train_images_count = sum([len(files) for r, d, files in os.walk('../input/landmark-recognition-2020/train')])

print('The number of train images is :', train_images_count)

test_images_count = sum([len(files) for r, d, files in os.walk('../input/landmark-recognition-2020/test')])

print('The number of test images is :', test_images_count)

print('The total number of images is :', train_images_count+test_images_count)

Base_path = '../input/landmark-recognition-2020/'

Train_DIR = f'{Base_path}/train'

Test_DIR = f'{Base_path}/test'

train = pd.read_csv(f'{Base_path}/train.csv')

submission = pd.read_csv(f'{Base_path}/sample_submission.csv')

print('Reading data completed')



samples = 20000

my_train_data = train.loc[:samples,:]

my_test_data = submission

my_train_data["filename"] = my_train_data.id.str[0]+"/"+my_train_data.id.str[1]+"/"+my_train_data.id.str[2]+"/"+my_train_data.id+".jpg"

my_train_data["label"] = my_train_data.landmark_id.astype(str)

print(samples ,' will be used in this notebook')
display(train.head())

print("Shape of train_data :", train.shape)
number_of_classes = len(my_train_data['landmark_id'].unique())

print('Number of unique classes in training images:',number_of_classes)

nb_images_pr_class= pd.DataFrame(train.landmark_id.value_counts())

nb_images_pr_class.reset_index(inplace=True)

nb_images_pr_class.columns = ['landmark_id','count']

print(nb_images_pr_class)

                

fig=plt.figure(figsize=(18, 3))

n = plt.hist(my_train_data["landmark_id"],bins=my_train_data["landmark_id"].unique())

#plt.ylim(top=250)

plt.title("Distribution of labels")

plt.xlabel("Landmark_id")

plt.ylabel("Number of images")

plt.show()
less_than_five = 0

between_five_and_ten = 0

for x in n[0]:

    if(x<5):

        less_than_five+=1

    elif(x<10):

        between_five_and_ten+=1

    

print('Number of classes that have less than 5 training samples :',less_than_five)

print('Number of classes that have between 5 and 10 training samples :',between_five_and_ten)
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')

plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(2, 2, figsize=(10, 8))



curr_row = 0

for i in range(4):

    example = cv2.imread(train_list[random.randint(0,len(train_list)-1)])

    example = example[:,:,::-1]

    

    col = i%2

    axarr[col, curr_row].imshow(example)

    if col == 1:

        curr_row += 1
plt.figure(figsize = (10, 8))

plt.title('Landmark ID Distribuition')

sns.distplot(my_train_data['landmark_id'])



plt.show()
nb_valid_samples = 0.2 # The percentage of the validation data

epochs = 10 # The maximum number of epochs

batch_size = 32 # The batch size

opt = 'RMSprop' # The used optimizer  

loss_function = 'categorical_crossentropy' # The loss function
#Network settings





gen = ImageDataGenerator(validation_split=nb_valid_samples)

# train_datagen = ImageDataGenerator(

#     rescale=1./255,

#     shear_range=0.2,

#     zoom_range=0.2,

#     horizontal_flip=True)



train_gen = gen.flow_from_dataframe(

    my_train_data,

    directory='../input/landmark-recognition-2020/train/',

    x_col='filename',

    y_col='label',

    weight_col=None,

    target_size=(256,256),

    color_mode='rgb',

    classes=None,

    class_mode='categorical',

    batch_size=batch_size,

    shuffle=True,

    subset='training',

    interpolation='nearest',

    validate_filenames=False)



val_gen = gen.flow_from_dataframe(

    my_train_data,

    directory='../input/landmark-recognition-2020/train/',

    x_col='filename',

    y_col='label',

    weight_col=None,

    target_size=(256,256),

    color_mode='rgb',

    classes=None,

    class_mode='categorical',

    batch_size=batch_size,

    shuffle=True,

    subset='validation',

    interpolation='nearest',

    validate_filenames=False)
model = tf.keras.Sequential([

    efn.EfficientNetB2(

        input_shape=(256, 256, 3),

        weights='imagenet',

        include_top=False

    ),

    GlobalAveragePooling2D(),

    Dense(number_of_classes, activation='softmax')

])



model.compile(opt, loss_function, metrics=['categorical_accuracy'])

model.summary()
train_steps = int(len(my_train_data)*(1-nb_valid_samples))//batch_size

val_steps = int(len(my_train_data)*nb_valid_samples)//batch_size



model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)





history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,

                              validation_data=val_gen, validation_steps=val_steps,

                              callbacks=[EarlyStopping(patience = 3, restore_best_weights = True),model_checkpoint])



model.save("model.h5")
print(history.history.keys())

plt.plot(history.history['categorical_accuracy'])

plt.plot(history.history['val_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
my_sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")

my_sub["filename"] = my_sub.id.str[0]+"/"+my_sub.id.str[1]+"/"+my_sub.id.str[2]+"/"+my_sub.id+".jpg"

print(my_sub)





best_model = load_model("best_model.h5")



test_gen = ImageDataGenerator().flow_from_dataframe(

    my_sub,

    directory="/kaggle/input/landmark-recognition-2020/test/",

    x_col="filename",

    y_col=None,

    weight_col=None,

    target_size=(256, 256),

    color_mode="rgb",

    classes=None,

    class_mode=None,

    batch_size=1,

    shuffle=True,

    subset=None,

    interpolation="nearest",

    validate_filenames=False)
my_sub
y_pred_one_hot = best_model.predict_generator(test_gen, verbose=1, steps=len(my_sub))
y_pred = np.argmax(y_pred_one_hot, axis=-1)

y_prob = np.max(y_pred_one_hot, axis=-1)

print(y_pred.shape, y_prob.shape)
y_uniq = np.unique(my_train_data.landmark_id.values)



y_pred = [y_uniq[Y] for Y in y_pred]
my_sub

for i in range(len(my_sub)):

    my_sub.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])

#my_sub = my_sub.drop(columns="filename")

my_sub.to_csv("submission.csv", index=False)

my_sub