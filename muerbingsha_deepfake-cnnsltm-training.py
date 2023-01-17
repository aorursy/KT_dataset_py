!pip install tqdm
from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()

print(len(devices))

for i in devices:

    print(i)
import tensorflow as tf

if tf.test.gpu_device_name():

    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:

    print("Please install GPU version of TF")
assert tf.test.is_gpu_available()

assert tf.test.is_built_with_cuda()
import tensorflow as tf 

import keras



config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 

sess = tf.Session(config=config) 

keras.backend.set_session(sess)
import glob 

import os 

import cv2 

import numpy as np 

import pandas as pd 

from tqdm.notebook import tqdm

import random

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from keras.layers import Conv3D, ConvLSTM2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling3D, Conv2D

from keras.layers import Input

from keras.models import Sequential, load_model, Model

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
SEQ_LENGTH = 10

IMG_SIZE = 150

BATCH_SIZE = 5

EPOCHS = 5

LABELS = ['REAL', 'FAKE']
dirs = glob.glob('dfdc*')



all_video_paths = np.array([])

all_labels = np.array([])





for d in tqdm(dirs):

    

    # get rid of all 4 trunks

    if d[-2:-1] == '4':

        continue



    meta = pd.read_json(d + '/metadata.json')

    vn = meta.columns.values

    vp = [d+'/'+n for n in vn]

    all_video_paths = np.concatenate([all_video_paths, vp])

    

    labels = [meta[n]['label'] for n in vn]

    labels = [LABELS.index(l) for l in labels]

    all_labels = np.concatenate([all_labels, labels])

    

# train_dir0 = 'dfdc_train_part_0/'

# train_dir1 = 'dfdc_train_part_1/'

# train_dir2 = 'dfdc_train_part_2/'



# meta0 = pd.read_json('dfdc_train_part_0/metadata.json')

# meta1 = pd.read_json('dfdc_train_part_1/metadata.json')

# meta2 = pd.read_json('dfdc_train_part_2/metadata.json')



# video_names0 = meta0.columns.values

# video_names1 = meta1.columns.values

# video_names2 = meta2.columns.values



# meta = meta0 + meta1 + meta2 

# video_names = meta.columns.values



# video_paths0 = [train_dir0 + name for name in video_names0]

# video_paths1 = [train_dir1 + name for name in video_names1]

# video_paths2 = [train_dir2 + name for name in video_names2]



# all_video_paths = video_paths0 + video_paths1 + video_paths2 

# print(all_video_paths[:2])





# tmp0 = [meta0[video_names0[i]].label for i in range(len(video_names0))]

# tmp1 = [meta1[video_names1[i]].label for i in range(len(video_names1))]

# tmp2 = [meta2[video_names2[i]].label for i in range(len(video_names2))]



# tmp = tmp0 + tmp1 + tmp2

# all_labels = [LABELS.index(l) for l in tmp]

# len(all_labels)

fake_index = np.where(np.array(all_labels) == 1)[0]

true_index = np.where(np.array(all_labels) == 0)[0]

print(len(fake_index))

print(len(true_index))
fake_index = np.random.choice(fake_index, len(true_index))

print(len(fake_index))
true_vp = [all_video_paths[i] for i in true_index]

true_label = [all_labels[i] for i in true_index]

fake_vp = [all_video_paths[i] for i in fake_index]

fake_label = [all_labels[i] for i in fake_index]

print(len(true_vp))

print(len(true_label))

print(len(fake_vp))

print(len(fake_label))



paths = np.concatenate([true_vp, fake_vp])

labels = np.concatenate([true_label, fake_label])

print(len(paths))

print(len(labels))
# shuffle 

c = list(zip(paths, labels))

random.shuffle(c)

paths, labels = zip(*c)
import seaborn as sns

sns.distplot(labels)
x_train, x_val, y_train, y_val = train_test_split(paths, labels, test_size=0.2)



print(len(x_train))

print(len(y_train))

print(len(x_val))

print(len(y_val))



print(x_train[:10])

print(y_train[:10])

print(x_val[:10])

print(y_val[:10])
def get_sequence(video_path):

    v_cap = cv2.VideoCapture(video_path)

    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    

    samples = np.linspace(0, v_len-1, SEQ_LENGTH).round()

    imgs = []

    for i in range(v_len):

        rel, frame = v_cap.read()

        if frame is None:

            print(video_path)

            

        if i in samples:

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # resize

            img = np.mean(img, axis=-1)

            img = np.expand_dims(img, axis=-1)

            imgs.append(img)

    

    # update i

    v_cap.release()

    

    return np.array(imgs)
# training

def generator():

    i = 0

    while True:

        # reset i

        if i >= len(x_train):

            i = 0  

        

        

        # get batch

        batch_videos = x_train[i:i+BATCH_SIZE]

        batch_labels = y_train[i:i+BATCH_SIZE]

        

        X = []

        for v in batch_videos:

            X.append(get_sequence(v))

            

            

        # upgrade i

        i += BATCH_SIZE

            

        yield np.array(X), np.array(batch_labels)
gen = generator()

a, b = next(gen)

print(a.shape)

print(b.shape)

gen = generator()
# Model 

def CNNLSTM(input_shape=(20, 300, 300, 1)):

    model = Sequential()



    model.add(Conv3D(16, kernel_size=3, activation='relu', padding='same', input_shape=input_shape,  data_format='channels_last'))

    model.add(BatchNormalization())

    

    model.add(ConvLSTM2D(32, kernel_size=(3), padding='same', return_sequences=True))

    model.add(BatchNormalization())

    

    model.add(Conv3D(1, kernel_size=3, activation='relu', padding='same', data_format='channels_last'))

    model.add(BatchNormalization())

    model.add(MaxPooling3D(strides=2))

    

    model.add(Conv3D(1, kernel_size=3, activation='relu', padding='same', data_format='channels_last'))

    model.add(BatchNormalization())

    model.add(MaxPooling3D(strides=2))

    

    model.add(Flatten())

    model.add(Dropout(0.5))

    

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(1, activation='sigmoid'))





    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))

    

    return model
# first train

model = CNNLSTM((SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 1))



# more train 

# model = load_model('./cnnlstm-5.h5')



model.summary()
# callbacks

es = EarlyStopping(monitor='loss', verbose=1, patience=3)

rc = ReduceLROnPlateau(monitor='loss', verbose=1, factor=0.2, patience=3, min_lr=1e-5)



cb = [es, rc]
# don't use hardcode video numbers

history = model.fit_generator(generator=gen, 

                              epochs=EPOCHS, 

                              steps_per_epoch = len(x_train)//BATCH_SIZE,

                              callbacks=cb)
# 

model.save('./cnnlstm-9.h5')