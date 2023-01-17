import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from keras.utils import Sequence

from keras import Model

from keras.optimizers import Adam

from keras.losses import mean_absolute_error, mean_squared_error
paths = []

for root, subdirs, files in os.walk('../input/osic-random-slices-from-lung-regions/image_slice'):

    for file in files:

        if os.path.splitext(file)[1].lower() in ('.npz'):

             paths.append(os.path.join(root, file))
class ImageGenerator(Sequence):

    def __init__(self, paths, batch_size=64, random_state=None):

        self._paths = paths

        self._batch_size = batch_size

        self._random = np.random.RandomState(random_state)

        

    def __len__(self):

        return len(self._paths) // self._batch_size

    

    def __getitem__(self, idx):

        paths = self._random.choice(self._paths, size=self._batch_size)

        imgs = np.expand_dims([np.load(p, allow_pickle=True)['arr_0'].tolist()['slice'] for p in paths], axis=-1) / 1_000

        return imgs, imgs
data_gen = ImageGenerator(paths, random_state=43)
batch = data_gen[0]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[0][8*i + j, ..., 0])

plt.show();
from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, AveragePooling2D, LeakyReLU

)



from tensorflow.keras import Model

from tensorflow.keras.optimizers import Nadam



def get_encoder(shape=(512, 512, 1)):

    def res_block(x, n_features):

        _x = x

        x = BatchNormalization()(x)

        x = LeakyReLU()(x)

    

        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

        x = Add()([_x, x])

        return x

    

    inp = Input(shape=shape)

    

    # 64

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 32

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 128)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 16

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 256)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 64

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(3):

        x = res_block(x, 128)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    

    # 8

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    for _ in range(2):

        x = res_block(x, 32)    

    

    # 8

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    return Model(inp, x)







def get_decoder(shape=(8, 8, 1)):

    inp = Input(shape=shape)



    # 8

    x = UpSampling2D((2, 2))(inp)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    # 16

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    # 32

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)



    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = BatchNormalization()(x)

    x = LeakyReLU()(x)

    

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    return Model(inp, x)
encoder = get_encoder((64, 64, 1))

decoder = get_decoder((4, 4, 1))



inp = Input((64, 64, 1))

e = encoder(inp)

d = decoder(e)

model = Model(inp, d)
encoder.summary()
decoder.summary()
model.compile(optimizer=Adam(lr=3*1e-3), loss='mae')

model.summary()
data_gen = ImageGenerator(paths, random_state=43, batch_size=512)

model.fit_generator(data_gen, steps_per_epoch=len(data_gen), epochs=5)
x_encode = encoder.predict_generator(data_gen, verbose=1)
e_samples, x_samples = [], []

for i in range(25):

    x, _ = data_gen[i]

    _x = encoder.predict(x)

    e_samples.extend(_x.tolist())

    x_samples.extend(x.tolist())
e_stack, x_stack = np.array(e_samples), np.array(x_samples)
x_encode = x_encode.reshape((-1, 16))
scaler = StandardScaler()

x_scale = scaler.fit_transform(x_encode)



pca = PCA()

x_pca = pca.fit_transform(x_scale)

sample_pca = pca.transform(scaler.transform(e_stack.reshape((-1, 16))))
kmeans = KMeans(n_clusters=10)

x_c = kmeans.fit_predict(x_pca)

sample_c = kmeans.predict(sample_pca.astype(x_pca.dtype))
plt.figure(figsize=(10,8))

plt.plot(1 - pca.explained_variance_ratio_);
plt.figure(figsize=(10,8))

plt.plot(x_pca[:, 0], x_pca[:, 1], '.', alpha=0.1);

plt.plot(sample_pca[:, 0], sample_pca[:, 1], '.', alpha=0.1);
import random

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(10)]



plt.figure(figsize=(10,8))



for c in range(10):

    subset = x_pca[x_c==c]

    plt.plot(subset[:, 0], subset[:, 1], '.', alpha=0.1, color=color[c]);
x = x_stack[sample_c == 0]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 1]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 2]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 3]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 4]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 5]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 6]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 7]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 8]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();
x = x_stack[sample_c == 9]



idx = np.random.randint(0, len(x), size=64)

batch = x[idx]



fig, ax = plt.subplots(8, 8, figsize=(16, 16))



for i, _ax in enumerate(ax):

    for j, __ax in enumerate(_ax):

        __ax.imshow(batch[8*i + j, ..., 0])

plt.show();