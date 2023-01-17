import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline



import tensorflow as tf

import tensorflow_hub as hub



import cv2

import os

print(os.listdir("../input"))
train_df = pd.read_csv("../input/flowerdataset/flowerDataset/flowers17_training.csv", header=None)
train_df.head()
y_train = train_df[0]

X_train = train_df.drop(columns=[0])
def get_sample_images(X_train, y_train):

    image_data = []

    labels = []

    print("Loading images for: ", end =" ")

    samples = np.random.choice(len(X_train), 16)

    for sample in samples:

        print("{} |".format(y_train.iloc[sample]), end=" ")

        img = X_train.iloc[sample].values.reshape((64,64,3))

        img = np.flip(img, 2)

        image_data.append(img)

        labels.append(y_train.iloc[sample])

        

    return np.array(image_data), labels
images, labels = get_sample_images(X_train, y_train)
def show_images(images, cols = 1, titles = None):

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: print('Serial title'); titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image, cmap=None)

        a.set_title(title, fontsize=50)

        a.grid(False)

        a.axis("off")

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

plt.show()
show_images(images, 4, titles=labels)
model = hub.load("https://github.com/captain-pool/GSOC/releases/download/1.0.0/esrgan.tar.gz")
init = tf.global_variables_initializer()
super_res = []

with tf.Session() as sess:

    sess.run(init)

    super_res.append(model.call(images).eval())

    fake_img = tf.cast(tf.clip_by_value(super_res[0], 0, 255), tf.uint8).eval()
fake_img.shape
show_images(images=fake_img, cols=4, titles=labels)