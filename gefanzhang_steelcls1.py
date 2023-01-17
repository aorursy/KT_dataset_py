!pip install -U efficientnet
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import math



# import os

# import tensorflow as tf

# import keras.backend.tensorflow_backend as ktf



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# config = tf.ConfigProto()

# config.gpu_options.per_process_gpu_memory_fraction = 0.9

# ktf.set_session(tf.Session(config=config))
trainset_dir = '../input/severstal-steel-defect-detection/train_images/'

trainset_csv = '../input/severstal-steel-defect-detection/train.csv'



trainset_raw_df = pd.read_csv(trainset_csv)

trainset_raw_df['dir'] = trainset_raw_df['ImageId_ClassId'].map(lambda x: trainset_dir + x.split('_')[0])



trainset_df = pd.DataFrame(columns=['dir', 'msk'])

trainset_list = []



for i in range(len(trainset_raw_df)):

    row = math.floor(i / 4)

    if i % 4 == 0:

        temp = {'dir': trainset_raw_df['dir'][i], 'msk': []}

    temp['msk'].append(trainset_raw_df['EncodedPixels'][i])

    if i % 4 == 3:

        trainset_list.append(temp)

trainset_df = trainset_df.append(trainset_list)
# build iterator

from keras.utils import Sequence

import cv2



class TrainDataFeeder(Sequence):



    def __init__(self, dataframe, batch_size):

        self.dataframe = dataframe

        self.batch_size = batch_size

        self.length = len(self.dataframe)

        self.dataframe_run = self.dataframe.sample(self.length).reset_index(drop=True)



    def __getitem__(self, index):

        start = index * self.batch_size

        end = min(self.length, (index + 1) * self.batch_size)

        x, y = [], []

        for i in range(start, end):

            if any(isinstance(ecd, str) for ecd in self.dataframe_run['msk'][i]):

                y_i = 1

            else:

                y_i = 0

            x_i = plt.imread(self.dataframe_run['dir'][i]).copy().astype(np.float32)

            x_i = cv2.resize(x_i, dsize=(0, 0), fx=0.5, fy=0.5)

            x_i -= np.mean(x_i, keepdims=True)

            x_i /= (np.std(x_i, keepdims=True) + 1e-6)

            x.append(x_i)

            y.append(y_i)

        return np.stack(x, axis=0), np.stack(y, axis=0)



    def __len__(self):

        return math.ceil(self.length / self.batch_size)



    def calculate_density(self):

        dense = [0, 0]

        for i in range(self.length):

            if any(isinstance(ecd, str) for ecd in self.dataframe_run['msk'][i]):

                dense[1] += 1

            else:

                dense[0] += 1

        return dense



    def on_epoch_end(self):

        self.dataframe_run = self.dataframe.sample(self.length).reset_index(drop=True)
bs = 20

valsplit = 0.05



trainset_df = trainset_df.sample(len(trainset_df)).reset_index(drop=True)



valcnt = int(len(trainset_df) * valsplit)

train_df = trainset_df[valcnt:].reset_index(drop=True)

val_df = trainset_df[:valcnt].reset_index(drop=True)



tdf = TrainDataFeeder(dataframe=train_df, batch_size=bs)

vdf = TrainDataFeeder(dataframe=val_df, batch_size=bs)



plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.bar(x=[1, 2], height=tdf.calculate_density(), width=0.5, align='center')

plt.subplot(1, 2, 2)

plt.bar(x=[1, 2], height=tdf.calculate_density(), width=0.5, align='center')



plt.show()
xb, yb = tdf[0]

plt.figure(figsize=(30, 30))

for i in range(bs):

    plt.subplot(bs, 1, i + 1)

    x = xb[i, :, :, :]

    x -= np.min(x)

    x /= np.max(x)

    plt.imshow(x)

print(yb)

plt.show()
# build the model

from keras.layers import *

from keras.models import *

from keras.optimizers import *

import keras.backend as K

from keras.utils import plot_model

from keras.metrics import categorical_accuracy

from efficientnet import EfficientNetB3



inputs = Input(shape=(128, 800, 3))

mdl = EfficientNetB3(

    input_tensor=inputs,

    input_shape=inputs.shape,

    include_top=False,

    pooling='avg'

)

outputs = mdl.layers[-1].output

outputs = Dense(units=1, activation='linear')(outputs)

outputs = Activation('sigmoid')(outputs)

mdl = Model(input=inputs, output=outputs)

mdl.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])



mdl.summary()

mdl.load_weights('../input/steelcls/weights.20-0.94.hdf5')

# plot_model(mdl, show_shapes=True, to_file='model.png')
# callbacks

from keras.callbacks import *



chkpnt = ModelCheckpoint(

    filepath='./weights.{epoch:02d}-{val_acc:.2f}.hdf5',

    monitor='val_acc',

    save_best_only=True,

    save_weights_only=False,

    mode='max',

    period=1,

    verbose=1,

)



lrrdcr = ReduceLROnPlateau(

    monitor='val_acc',

    mode='max',

    factor=0.1,

    patience=3,

    verbose=1

)
# train

mdl.fit_generator(

    epochs=20,

    generator=tdf,

    steps_per_epoch=len(tdf),

    validation_data=vdf,

    validation_steps=len(vdf),

    shuffle=False,

    callbacks=[chkpnt, lrrdcr]

)