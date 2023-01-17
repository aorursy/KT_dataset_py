import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Dropout, GlobalAveragePooling2D

from keras.models import Model, Sequential

from keras.applications.vgg19 import VGG19 
base_dir = '/kaggle/input/bengaliai-cv19'

# for f in os.listdir(base_dir):

#     print(f)

train_folders = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.startswith('train_image_data')])

train_csv = os.path.join(base_dir, 'train.csv')

class_map = os.path.join(base_dir, 'class_map.csv')
train_df = pd.read_csv(train_csv)

df_class = pd.read_csv(class_map)
# grapheme, g_counts = np.unique(df['grapheme_root'], return_counts=True)

# vowel, v_counts = np.unique(df['vowel_diacritic'], return_counts=True)

# consonant, c_counts = np.unique(df['consonant_diacritic'], return_counts=True)

# fig = plt.figure(figsize=(20, 20))

# plt.scatter(consonant, c_counts, s=c_counts, c=c_counts)

# plt.show()
df0 = pd.read_parquet(train_folders[0])

df1 = pd.read_parquet(train_folders[1])

df2 = pd.read_parquet(train_folders[2])

df3 = pd.read_parquet(train_folders[3])
df0 = df0.iloc[:, 1:]

df1 = df1.iloc[:, 1:]

df2 = df2.iloc[:, 1:]

df3 = df3.iloc[:, 1:]
df = pd.concat([df0, df1, df2, df3])

del [df0, df1, df2, df3]
WIDTH = 137

HEIGHT = 236

BATCH_SIZE = 32
class BengaliGenerator(keras.utils.Sequence):

    def __init__(self ,data, batch_size, dim):

        self.data = data

        self.labels1 = pd.get_dummies(data['grapheme_root'])

        self.labels2 = pd.get_dummies(data['vowel_diacritic'])

        self.labels3 = pd.get_dummies(data['consonant_diacritic'])

        self.batch_size = batch_size

        self.dim = dim

        self.list_ids = self.data.index.values

        self.on_epoch_end()

        

    def __len__(self):

        return int(np.floor(len(self.data) / self.batch_size))

    

    def __getitem__(self,idx):

        imgs = df.iloc[idx*self.batch_size: (idx+1)*self.batch_size].values.reshape(-1, WIDTH, HEIGHT)

        

#         imgs = np.expand_dims(imgs, axis=3)

#         for i in range(len(imgs)):

#             for j in range(1, 3):

#                 imgs[i, :, :, j] = imgs[i, :, :, 0]



        imgs = np.repeat(imgs[..., np.newaxis], 3, -1)

        

        labels0 = pd.get_dummies(self.data['grapheme_root'])

        labels1 = pd.get_dummies(self.data['vowel_diacritic'])

        labels2 = pd.get_dummies(self.data['consonant_diacritic'])

        y0 = labels0.iloc[idx*self.batch_size: (idx+1)*self.batch_size].values

        del labels0

        y1 = labels1.iloc[idx*self.batch_size: (idx+1)*self.batch_size].values

        del labels1

        y2 = labels2.iloc[idx*self.batch_size: (idx+1)*self.batch_size].values

        del labels2



        return imgs, [y0, y1, y2]

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_ids))

#         if self.shuffle:

#             np.random.shuffle(self.indexes)
train, val = train_test_split(train_df, test_size = 0.2, random_state = 2019)
train_gen = BengaliGenerator(train, BATCH_SIZE, (WIDTH, HEIGHT ))

val_gen = BengaliGenerator(val, BATCH_SIZE, (WIDTH, HEIGHT))
conv_base = VGG19(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT, 3))
for layer in conv_base.layers[:-3]:

    layer.trainable = False
# conv_base.summary()
inp = Input(shape = (WIDTH, HEIGHT, 3))

output = conv_base(inp)

x = GlobalAveragePooling2D()(output)

out1 = Dense(168, activation = 'softmax')(x)

out2 = Dense(11, activation = 'softmax')(x)

out3 = Dense(7, activation = 'softmax')(x)

    

model = Model(inputs = inp, outputs = [out1,out2,out3])

model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
# history = model.fit_generator(train_gen, epochs=10, steps_per_epoch=len(train) / BATCH_SIZE)

# #                               ,validation_data=val_gen, validation_steps=len(val) / BATCH_SIZE)