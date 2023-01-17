# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/all-dogs/"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Activation, Reshape

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Flatten, Dropout

from keras.preprocessing.image import img_to_array, load_img

from keras.optimizers import Adam

import math

import numpy as np

import os

from tqdm import tqdm

from PIL import Image

from keras.preprocessing import image
img_list =os.listdir("../input/all-dogs/all-dogs/")
print(os.listdir("../working"))
len(img_list)
temp_img = load_img('../input/all-dogs/all-dogs/n02085620_10074.jpg')

temp_img_array  = img_to_array(temp_img)
temp_img
temp_img_array.shape
def generator_model():

    model = Sequential()

    model.add(Dense(input_dim=100, units=1024))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dense(32 * 32 * 128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Reshape((32, 32, 128), input_shape=(32 * 32 * 128,)))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(64, (5, 5), padding="same"))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(3, (5, 5), padding="same"))

    model.add(Activation('tanh'))

    return model
def discriminator_model():

    model = Sequential()

    model.add(Conv2D(64, (5,5), strides=(2, 2), input_shape=(128, 128, 3), padding="same"))

    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (5,5), strides=(2, 2)))

    model.add(LeakyReLU(0.2))

    model.add(Flatten())

    model.add(Dense(256))

    model.add(LeakyReLU(0.2))

    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.add(Activation('sigmoid'))

    return model
def combine_images(generated_images):

    total = generated_images.shape[0]

    cols = int(math.sqrt(total))

    rows = math.ceil(float(total)/cols)

    width, height, ch= generated_images.shape[1:]

    output_shape = (

        height * rows,

        width * cols,

        ch

    )

    combined_image = np.zeros(output_shape)



    for index, image in enumerate(generated_images):

        i = int(index/cols)

        j = index % cols

        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, :]

    return combined_image
TRAIN_IMAGE_PATH = '../input/all-dogs/all-dogs/'

#GENERATED_IMAGE_PATH = '../images/'

GENERATED_IMAGE_PATH = '../working/images/'

GEN_GENERATED_IMAGE_PATH = '../gen_images/'
# 訓練データ読み込み

img_list = os.listdir(TRAIN_IMAGE_PATH)

X_train = []

for img in img_list:

    img = img_to_array(load_img(TRAIN_IMAGE_PATH+img, target_size=(128,128,3)))

    # -1から1の範囲に正規化

    img = (img.astype(np.float32) - 127.5)/127.5

    X_train.append(img)
len(X_train)
# 4Dテンソルに変換(データの個数, 128, 128, 3)

X_train = np.array(X_train)
# generatorとdiscriminatorを作成

discriminator = discriminator_model()

d_opt = Adam(lr=1e-5, beta_1=0.1)

discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

# discriminatorの重みを固定(dcganの中のみ)

discriminator.trainable = False

generator = generator_model()



dcgan = Sequential([generator, discriminator])

g_opt = Adam(lr=2e-4, beta_1=0.5)

dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

BATCH_SIZE = 128

NUM_EPOCH  = 200



num_batches = int(X_train.shape[0] / BATCH_SIZE)

print('Number of batches:', num_batches)

for epoch in tqdm(range(NUM_EPOCH)):

    for index in range(num_batches):

        noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])

        image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

        generated_images = generator.predict(noise, verbose=0, batch_size=BATCH_SIZE)



#         # 生成画像を出力

#         if (index+1) % (num_batches) == 0:

#             image = combine_images(generated_images)

#             image = image*127.5 + 127.5

#             if not os.path.exists(GENERATED_IMAGE_PATH):

#                 os.mkdir(GENERATED_IMAGE_PATH)

#             Image.fromarray(image.astype(np.uint8)).save(GENERATED_IMAGE_PATH+"%04d_%04d.png" % (epoch, index))

            

        if not os.path.exists(GEN_GENERATED_IMAGE_PATH):

            os.mkdir(GEN_GENERATED_IMAGE_PATH)

        

        if epoch == 200 and index > 59:

            generated_images_v = generated_images*127.5 + 127.5    

            for j in range(100):

                Image.fromarray((generated_images_v[j]*127.5 + 127.5).astype(np.uint8)).save(GEN_GENERATED_IMAGE_PATH+"%04d_%04d_%04d.png" % (epoch, index,j))



        # discriminatorを更新

        X = np.concatenate((image_batch, generated_images))

        # 訓練データのラベルが1、生成画像のラベルが0になるよう学習する

        y = [1]*BATCH_SIZE + [0]*BATCH_SIZE

        d_loss = discriminator.train_on_batch(X, y)



        # generator更新

        noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])

        # 生成画像をdiscriminatorにいれたときに

        # 出力が1に近くなる(訓練画像と識別される確率が高くなる)ように学習する

        g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)



        print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss))
print(os.listdir(GENERATED_IMAGE_PATH))
import shutil

#shutil.make_archive('images', 'zip', '../images/')

shutil.make_archive('images', 'zip', '../gen_images/')