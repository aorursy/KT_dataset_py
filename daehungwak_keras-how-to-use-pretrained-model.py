import gc

import os

import glob

import zipfile

import warnings

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm



import cv2

import PIL

from PIL import ImageOps, ImageFilter, ImageDraw
DATA_PATH = '../input/'

os.listdir(DATA_PATH)
# 이미지 폴더 경로

TRAIN_CROP_PATH = "./train_crop"

TEST_CROP_PATH = "./test_crop"

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')

TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')



# CSV 파일 경로

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))
df_train.head()
df_test.head()
def crop_resize_boxing_img(img_name, margin=16, size=(224, 224)) :

    if img_name.split('_')[0] == "train" :

        PATH = TRAIN_IMG_PATH

        data = df_train

    elif img_name.split('_')[0] == "test" :

        PATH = TEST_IMG_PATH

        data = df_test

        

    img = PIL.Image.open(os.path.join(PATH, img_name))

    pos = data.loc[data["img_file"] == img_name, \

                   ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)



    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)



    return img.crop((x1,y1,x2,y2)).resize(size)
!mkdir {TRAIN_CROP_PATH}
%%time

for i, row in df_train.iterrows():

    cropped = crop_resize_boxing_img(row['img_file'])

    cropped.save(f"{TRAIN_CROP_PATH}/{row['img_file']}")
!mkdir {TEST_CROP_PATH}
%%time

for i, row in df_test.iterrows():

    cropped = crop_resize_boxing_img(row['img_file'])

    cropped.save(f"{TEST_CROP_PATH}/{row['img_file']}")
tmp_imgs = df_train['img_file'][100:105]

plt.figure(figsize=(12,20))



for num, f_name in enumerate(tmp_imgs):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, f_name))

    plt.subplot(5, 2, 2*num + 1)

    plt.title(f_name)

    plt.imshow(img)

    plt.axis('off')

    

    img_crop = PIL.Image.open(f"train_crop/{f_name}")

    plt.subplot(5, 2, 2*num + 2)

    plt.title(f_name + ' cropped')

    plt.imshow(img_crop)

    plt.axis('off')
from sklearn.model_selection import train_test_split



df_train["class"] = df_train["class"].astype('str')



df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]



its = np.arange(df_train.shape[0])

train_idx, val_idx = train_test_split(its, train_size = 0.8, random_state=42)



X_train = df_train.iloc[train_idx, :]

X_val = df_train.iloc[val_idx, :]



print(X_train.shape)

print(X_val.shape)

print(df_test.shape)
import tensorflow as tf

from tensorflow.keras.applications.mobilenet import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Parameter

img_size = (224, 224)

nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

nb_test_samples = len(df_test)

epochs = 20

batch_size = 32



# Define Generator config

train_datagen = ImageDataGenerator(

    horizontal_flip = True, 

    vertical_flip = False,

    preprocessing_function=preprocess_input

)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



# Make Generator

train_generator = train_datagen.flow_from_dataframe(

    dataframe=X_train, 

    directory=TRAIN_CROP_PATH,

    x_col = 'img_file',

    y_col = 'class',

    target_size = img_size,

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    seed=42

)



validation_generator = val_datagen.flow_from_dataframe(

    dataframe=X_val, 

    directory=TRAIN_CROP_PATH,

    x_col='img_file',

    y_col='class',

    target_size=img_size,

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    shuffle=False

)



test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

    directory=TEST_CROP_PATH,

    x_col='img_file',

    y_col=None,

    target_size= img_size,

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False

)
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D
# for layer in resNet_model.layers:

#     layer.trainable = False

#     print(layer,layer.trainable)



mobileNetModel = MobileNet(weights='imagenet', include_top=False)



model = Sequential()

model.add(mobileNetModel)

model.add(GlobalAveragePooling2D())

model.add(Dense(196, activation='softmax', kernel_initializer='he_normal'))

model.summary()
from sklearn.metrics import f1_score



def micro_f1(y_true, y_pred):

    return f1_score(y_true, y_pred, average='micro')



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
def get_steps(num_samples, batch_size):

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
%%time

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



filepath = "my_mobilenet_model_{val_acc:.2f}_{val_loss:.4f}.h5"



ckpt = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

es = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')



callbackList = [ckpt]



history = model.fit_generator(

    train_generator,

    steps_per_epoch = get_steps(nb_train_samples, batch_size),

    epochs=epochs,

    validation_data = validation_generator,

    validation_steps = get_steps(nb_validation_samples, batch_size),

    callbacks = callbackList

)

gc.collect()
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model_list = sorted([i for i in os.listdir() if "my_" in i])

model_list
model.load_weights(model_list[-1])
%%time

test_generator.reset()

prediction = model.predict_generator(

    generator = test_generator,

    steps = get_steps(nb_test_samples, batch_size),

    verbose=1

)
predicted_class_indices=np.argmax(prediction, axis=1)



# Generator class dictionary mapping

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission["class"] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()
!rm -rf *_crop