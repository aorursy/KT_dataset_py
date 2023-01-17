from skimage import io, transform

import os

import xml.etree.ElementTree  as ET

from matplotlib import pyplot as plt

from tqdm import tqdm, tqdm_notebook

import gc

from keras.layers import Input, Dropout, Dense, Embedding, Flatten, Activation, BatchNormalization, Reshape, UpSampling2D

from keras.optimizers import Adam

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.models import Model

from keras.layers.advanced_activations import LeakyReLU

from keras.initializers import RandomNormal

import numpy as np
def get_images():

    base_dir = "../input/annotation/Annotation/"

    img_dir = "../input/all-dogs/all-dogs/"

    paths = []

    imgs = []

    ret = os.listdir(base_dir)

    for x in ret:

        for y in os.listdir(base_dir + x):

            paths.append(base_dir + x + "/" + y)



    for x in tqdm_notebook(paths):

        try:

            tree = ET.ElementTree()

            root = tree.parse(x)

            img_path = img_dir + root.find("filename").text + ".jpg"

            x1 = int(root.find("object").find("bndbox").find("xmin").text)

            y1 = int(root.find("object").find("bndbox").find("ymin").text)

            x2 = int(root.find("object").find("bndbox").find("xmax").text)

            y2 = int(root.find("object").find("bndbox").find("ymax").text)

            img = io.imread(img_path)

            img = img[y1:y2 + 1, x1:x2 + 1, :]

            width = x2 - x1 + 1

            height = y2 - y1 + 1

            if height > width:

                width = int(64.0 * width / height)

                height = 64

                img = transform.resize(img, output_shape=(height, width, 3), preserve_range=True) / 255.0

                img = np.pad(img, ((0, 0), ((64 - width) // 2, (64 - width) - (64 - width) // 2), (0, 0)), 

                       mode="constant", constant_values=0.0)

            else:

                height = int(64.0 * height / width)

                width = 64

                img = transform.resize(img, output_shape=(height, width, 3), preserve_range=True) / 255.0

                img = np.pad(img, (((64 - height) // 2, (64 - height) - (64 - height) // 2), (0, 0), (0, 0)), 

                       mode="constant", constant_values=0.0)

            img = (img - 0.5) * 2

            imgs.append(img)

        except:

            pass

    

    del paths

    return imgs
imgs = get_images()

gc.collect()
def generator():

    init = RandomNormal(stddev=0.02)

    z = Input(shape=(100,), name="noise")

    x = Dense(4 * 4 * 64, kernel_initializer=init)(z)

    x = Reshape(target_shape=(4, 4, 64))(x)

    x = Conv2D(128, 3, activation="relu", kernel_initializer=init, padding="same")(UpSampling2D()(x))

    x = Conv2D(128, 3, activation="relu", kernel_initializer=init, padding="same")(UpSampling2D()(x))

    x = Conv2D(128, 3, activation="relu", kernel_initializer=init, padding="same")(UpSampling2D()(x))

    x = Conv2D(128, 3, activation="relu", kernel_initializer=init, padding="same")(UpSampling2D()(x))

    x = Conv2D(3, 3, activation="tanh", kernel_initializer=init, padding="same")(x)

    model = Model(z, x)

    return model
model = generator()

model.summary()
def discrimator():

    init = RandomNormal(stddev=0.02)

    img = Input(shape=(64, 64, 3), name="image")

    x = Dropout(0.25)(LeakyReLU(0.2)(Conv2D(128, 3, strides=(2, 2), 

                                            padding="same", kernel_initializer=init)(img)))

    x = Dropout(0.25)(LeakyReLU(0.2)(Conv2D(128, 3, strides=(2, 2), 

                                            padding="same", kernel_initializer=init)(x)))

    x = Dropout(0.25)(LeakyReLU(0.2)(Conv2D(128, 3, strides=(2, 2), 

                                            padding="same", kernel_initializer=init)(x)))

    x = Dropout(0.25)(LeakyReLU(0.2)(Conv2D(128, 3, strides=(2, 2), 

                                            padding="same", kernel_initializer=init)(x)))

    x = Flatten()(x)

    x = Dense(1, kernel_initializer=init, activation="sigmoid")(x)

    model = Model(img, x)

    return model
model = discrimator()

model.summary()
def train(imgs, epochs, batch_size=4):

    d = discrimator()

    d.compile(loss='binary_crossentropy',

              optimizer=Adam(0.0006, 0.5), 

              metrics=['accuracy'])

    

    d.trainable = False

    g = generator()

    z = Input(shape=(100,), name="noise")

    x = d(g(z))

    combine = Model(z, x)

    combine.compile(loss='binary_crossentropy',

                    optimizer=Adam(0.0004, 0.5),

                    metrics=['accuracy'])

    

    d.summary()

    combine.summary()

    

    true_label = np.ones(shape=(batch_size, 1))

    false_label = np.zeros(shape=(batch_size, 1))

    

    for _ in range(epochs):

        index = len(imgs) // batch_size

        for idx in range(index):

            smooth = np.random.uniform(0, 0.1)

            img_in = np.array(imgs[idx * batch_size : (idx + 1) * batch_size])

            noise = np.random.randn(batch_size,100)

            gen_imgs = g.predict(noise)

          

            # 训练D, 使D能分辨真假

            d.trainable = True

            real_loss = d.train_on_batch(x=img_in, y=true_label * (1 - smooth))

            fake_loss = d.train_on_batch(x=gen_imgs, y=false_label)

            d_loss = (real_loss[0] + fake_loss[0]) / 2

            d_accu = (real_loss[1] + fake_loss[1]) / 2

            

            # 训练G, 使G往真的方向走, 因为noise是相同的, 所以和D训练的样本是一致的

            d.trainable = False

            g_loss = combine.train_on_batch(x=noise, y=true_label * (1 - smooth))

            

            if idx % 100 == 0:

                print("idx:", idx, "d_loss:", d_loss, "g_loss:", g_loss[0], "d_accu:", d_accu)

                

    return d, g
imgs = imgs[:10000]

np.random.shuffle(imgs)

d, g = train(imgs=imgs, epochs=200, batch_size=32)
noise = np.random.uniform(-1.0, 1.0, [10000, 100])

gen_imgs = g.predict(noise)

gen_imgs = (gen_imgs + 1) / 2

gen_imgs = (gen_imgs * 255).astype(np.uint8)

if not os.path.exists("result"):

    os.makedirs("result")

for i in range(10000):

    io.imsave("result/" + str(i) + ".png", gen_imgs[i])

import shutil

shutil.make_archive('images', 'zip', 'result')

shutil.rmtree('result')