from skimage import io, transform

import os

import xml.etree.ElementTree  as ET

from matplotlib import pyplot as plt

from tqdm import tqdm, tqdm_notebook

import gc

from keras.layers import Lambda, Input, Dropout, Dense, Embedding, Flatten, Activation, BatchNormalization, Reshape, UpSampling2D

from keras.optimizers import Adam

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.models import Model

from keras.layers.advanced_activations import LeakyReLU

from keras.initializers import RandomNormal

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
def get_images(padding=True):

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

            

            if padding:

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

            else:

                img = transform.resize(img, output_shape=(64, 64, 3), preserve_range=True) / 255.0

            imgs.append(img[np.newaxis, :, :, :])

        except:

            pass

    

    del paths

    

    return imgs
imgs = get_images(padding=False)
def generator():

    init = RandomNormal(stddev=0.02)

    z = Input(shape=(256,), name="noise")

    x = Reshape(target_shape=(1, 1, 256))(z)

    x = Activation("relu")(

        BatchNormalization(momentum=0.1)(

            Conv2DTranspose(1024, 4, kernel_initializer=init, use_bias=False)(x)

        )

    )

    x = Activation("relu")(

        BatchNormalization(momentum=0.1)(

            Conv2DTranspose(512, 4, strides=(2, 2), padding="same",

                            kernel_initializer=init, use_bias=False)(x)

        )

    )

    x = Activation("relu")(

        BatchNormalization(momentum=0.1)(

            Conv2DTranspose(256, 4, strides=(2, 2), padding="same",

                            kernel_initializer=init, use_bias=False)(x)

        )

    )

    x = Activation("relu")(

        BatchNormalization(momentum=0.1)(

            Conv2DTranspose(128, 4, strides=(2, 2), padding="same",

                            kernel_initializer=init, use_bias=False)(x)

        )

    )

    x = Activation("relu")(

        BatchNormalization(momentum=0.1)(

            Conv2DTranspose(64, 4, strides=(2, 2), padding="same",

                            kernel_initializer=init, use_bias=False)(x)

        )

    )

    x = Activation("tanh")(

        Conv2DTranspose(3, 3, padding="same", kernel_initializer=init)(x)

    )

    model = Model(z, x)

    return model
def discrimator():

    init = RandomNormal(stddev=0.02)

    img = Input(shape=(64, 64, 3), name="image")

    x = LeakyReLU(0.2)(Conv2D(32, 4, strides=(2, 2), 

                              padding="same", kernel_initializer=init)(img))

    x = LeakyReLU(0.2)(Conv2D(64, 4, strides=(2, 2), 

                              padding="same", kernel_initializer=init)(x))

    x = LeakyReLU(0.2)(

        BatchNormalization(momentum=0.1)(

             Conv2D(128, 4, strides=(2, 2), 

                    padding="same", kernel_initializer=init,

                    use_bias=False)(x)

        )

       )

    x = LeakyReLU(0.2)(

        BatchNormalization(momentum=0.1)(

             Conv2D(256, 4, strides=(2, 2), 

                    padding="same", kernel_initializer=init,

                    use_bias=False)(x)

        )

    )

    x = Conv2D(1, 4, kernel_initializer=init, use_bias=False)(x)

    x = Flatten()(x)

    model = Model(img, x)

    return model
def descrimator_combine(g, d):

    g.trainable = False

    d.trainable = True

    

    z = Input(shape=(256,), name="noise")

    img = Input(shape=(64, 64, 3), name="image")

    x1 = d(img)

    x2 = d(g(z))

    

    out1 = Lambda(lambda x: x[0] - K.mean(x[1]))([x1, x2])

    out2 = Lambda(lambda x: K.mean(x[0]) - x[1])([x1, x2])

    model = Model([z, img], [out1, out2])

    return model

    



def generator_combine(g, d):

    g.trainable = True

    d.trainable = False

    

    z = Input(shape=(256,), name="noise")

    img = Input(shape=(64, 64, 3), name="image")

    x1 = d(img)

    x2 = d(g(z))

    

    out1 = Lambda(lambda x: x[1] - K.mean(x[0]))([x1, x2])

    out2 = Lambda(lambda x: K.mean(x[1]) - x[0])([x1, x2])

    model = Model([z, img], [out1, out2])

    return model
def scale_cos(start, end, x):

    return start + (1 + np.cos(np.pi * (1 - x))) * (end - start) / 2



	

def rate_decay(x):

	if x < 0.2:

		return scale_cos(0.00001, 0.0004, x / 0.2)

	else:

		return scale_cos(0.0004, 0.00001, (x - 0.2)/ 0.8) 

    

x = list(map(lambda x: x / 10000.0, list(range(0, 10000))))

plt.plot(x, list(map(rate_decay, x)))

plt.show()
def train(imgs, epochs, batch_size=4):

    d = discrimator()

    g = generator()

    d_model = descrimator_combine(g, d)

    d_model.compile(loss="mse", optimizer=Adam(0.0001, 0.5),

                   loss_weights=[0.5, 0.5])

    g_model = generator_combine(g, d)

    g_model.compile(loss="mse", optimizer=Adam(0.0001, 0.5),

                   loss_weights=[0.5, 0.5])

    d_model.summary()

    g_model.summary()

    

    true_label = np.ones(shape=(batch_size, 1))

    

    datagen = ImageDataGenerator(

        rotation_range=5,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest'

    )



    idx = 0

    for _ in range(epochs):

        gc.collect()

        index = len(imgs) // batch_size

        count = 0

        for img_in in datagen.flow(x=np.vstack(imgs), batch_size=batch_size, shuffle=True):

            count += 1

            if count == index:

                break

            smooth = np.random.uniform(0, 0.1)

            img_in = (img_in - 0.5) * 2

            noise = np.random.randn(batch_size, 256)

          

            d_loss = d_model.train_on_batch(x=[noise, img_in],

                                         y=[true_label * (1 - smooth), true_label * (1 - smooth)])

            

            for _ in range(1):

                g_loss = g_model.train_on_batch(x=[noise, img_in], 

                                             y=[true_label * (1 - smooth), true_label * (1 - smooth)])

            

            if idx % 100 == 0:

                print("idx:", idx, "d_loss:", d_loss, "g_loss:", g_loss)

            idx += 1

            

            K.set_value(d_model.optimizer.lr, rate_decay(idx * 1.0 / (index * epochs)))

            K.set_value(g_model.optimizer.lr, rate_decay(idx * 1.0 / (index * epochs)))

                

    return d, g
np.random.shuffle(imgs)

d, g = train(imgs=imgs, epochs=300, batch_size=32)
from scipy.stats import truncnorm



def truncated_normal(size, threshold=1):

    values = truncnorm.rvs(-threshold, threshold, size=size)

    return values



noise = truncated_normal((10000, 256), 3)

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