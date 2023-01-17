import os

import glob

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

from IPython.display import display,Image

from tqdm.notebook import tqdm

import random

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

import random

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# !ls
# os.makedirs('images',exist_ok=True)
base_folder_input = '/kaggle/input/bengaliai-cv19/'

# train_files = glob.glob(base_folder_input+'train_image_data_*')

# test_files = glob.glob(base_folder_input+'test_image_data_*')

# print(len(train_files),len(test_files))

base_folder = '/kaggle/input/bengali-character-npz-files/images/'

# os.makedirs(base_folder,exist_ok=True)

# def save_data(files,folder):

#     os.makedirs(base_folder+folder,exist_ok=True)

#     for idx in tqdm(range(len(files))):

#         df = pd.read_parquet(files[idx], engine='pyarrow')

#         for index in tqdm(range(len(df))):

#             img_name = df.iloc[index]['image_id']

#             image = list(df.loc[df.index[index]].values[1:])

#             data = np.asarray(image)

#             # save to npy file

#             np.savez_compressed(base_folder+folder+'/'+img_name+'.npz', data)

#         del df

# #             break

# #         break

# #     pass

# save_data(train_files,'train')

# save_data(test_files,'test')
images = glob.glob(base_folder+'train/*')

print(len(images))
images = glob.glob(base_folder+'train/*')

random.shuffle(images)

train_files, val_files = train_test_split(images, test_size=0.33, random_state=42)

print(len(train_files),len(val_files),len(images))
df_train = pd.read_csv(base_folder_input+'train.csv')

df_train.head()
def image_generator(train_files,df_train,batch_size):

    count = 0

    x = []

    y1 = []

    y2 = []

    y3 = []

    while True:

        idx = random.choice(range(0,len(train_files)))

        

        

        img_name = train_files[idx].split("/")[-1][:-4]

        image = np.load(train_files[idx])['arr_0'].reshape(137,236)

        temp_df = df_train[df_train['image_id']==img_name]

        y_grapheme = np.zeros(168)

        y_grapheme[temp_df.iloc[0]['grapheme_root']] = 1

        y_vowel = np.zeros(11)

        y_vowel[temp_df.iloc[0]['vowel_diacritic']] = 1

        y_consonant = np.zeros(7)

        y_consonant[temp_df.iloc[0]['consonant_diacritic']]=1

        

        

        x.append(np.uint8(image))

        y1.append(y_grapheme)

        y2.append(y_vowel)

        y3.append(y_consonant)

        count += 1

        

        if count % batch_size == 0:

            grayscale_batch = np.array(x)

            rgb_batch = np.repeat(grayscale_batch[..., np.newaxis], 3, -1)

            yield rgb_batch, [np.array(y1),np.array(y2),np.array(y3)]

            x = []

            y1 = []

            y2 = []

            y3 = []

            count = 0
batch_size=4

learning_rate=3e-4

train_gen = image_generator(train_files,df_train,batch_size)

val_gen = image_generator(val_files,df_train,batch_size)

from keras.applications import ResNet50, MobileNet, Xception, DenseNet121, InceptionV3

# from keras.layers import GlobalAveragePooling2D

# from keras.callbacks import Callback

from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

from keras.preprocessing.image import ImageDataGenerator

from keras.backend import clear_session

from keras.models import Model, load_model

from keras.layers import Dense, Input, Flatten,GlobalAveragePooling2D

from keras.optimizers import adam
base_model = ResNet50(weights='imagenet', include_top=False)

model_name = 'ResNet50'

# base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes

predictions_grapheme_root = Dense(168, activation='softmax',name="grapheme_root")(x)

predictions_vowel_diacritic = Dense(11, activation='softmax',name="vowel_diacritic")(x)

predictions_consonant_diacritic = Dense(7, activation='softmax',name="consonant_diacritic")(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=[predictions_grapheme_root,predictions_vowel_diacritic,predictions_consonant_diacritic])

losses = {"grapheme_root": "categorical_crossentropy","vowel_diacritic": "categorical_crossentropy","consonant_diacritic":"categorical_crossentropy"}

cust_adam = adam(lr=learning_rate)

model.compile(optimizer=cust_adam,loss=losses, metrics=['accuracy'])
checkpoint_filepath = model_name+'.hdf5'

checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# early_stop = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=3, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=2e-7, mode='min', verbose=1)

csv_logger = CSVLogger(filename=model_name+'_log.csv')

callbacks_list = [checkpoint, reduce_lr,  csv_logger]





history = model.fit_generator(

            train_gen,

            steps_per_epoch=100,

            epochs=2,

            validation_data=val_gen,

            validation_steps=80,

            callbacks=callbacks_list

        )
from keras.models import load_model

model = load_model('ResNet50.hdf5')
test_files = glob.glob(base_folder+'test/*')

print(len(test_files))
testX = []

for idx in range(len(test_files)):

    img_name = test_files[idx].split("/")[-1][:-4]

    image = np.load(test_files[idx])['arr_0'].reshape(137,236)

    testX.append(np.uint8(image))

grayscale_batch = np.array(testX)

rgb_batch = np.repeat(grayscale_batch[..., np.newaxis], 3, -1)
MEAN = np.mean(rgb_batch, axis=(0, 1, 2))

STD = np.std(rgb_batch, axis=(0, 1, 2))

#     print(valX.shape)

for i in range(3):

    rgb_batch[:, :, :, i] = (rgb_batch[:, :, :, i] - MEAN[i]) / STD[i]
predictions = model.predict(rgb_batch)
dict_= {"row_id":[],"target":[]}

for idx in range(len(test_files)):

    image_id = test_files[idx].split("/")[-1][:-4]

    for key,value in {0:"grapheme_root",1:"vowel_diacritic",2:"consonant_diacritic"}.items():

        name  = image_id+'_'+value

        val = np.argmax(predictions[key][idx])

        dict_['row_id'].append(name)

        dict_['target'].append(val)

result_df = pd.DataFrame(dict_)

result_df.to_csv('submission.csv',index=False)
result_df.head()