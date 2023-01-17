from fastai.vision import *   # Quick access to computer vision functionality

import numpy as np

import os
# http://playground.tensorflow.org

# https://github.com/fastai/fastai/blob/master/courses/ml1/lesson4-mnist_sgd.ipynb
# Download our data

path = untar_data(URLs.MNIST_SAMPLE)

os.listdir(path)
# Let's load our data!

data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)

data.normalize(imagenet_stats)

pass
learn = cnn_learner(data, models.resnet18, metrics=accuracy)

learn.fit_one_cycle(1, 0.01)
image_index = 8
img,label = data.train_ds[image_index]

img = img.data.numpy().transpose(1, 2, 0)

fig = plt.figure(figsize=(6, 6))

plt.imshow(img)

plt.show()

print('We are looking at image number', image_index, 'from training dataset')
img, _ = data.train_ds[image_index]

learn.predict(img)[0]
image_index = 8

img,label = data.train_ds[image_index]

img = img.data.numpy().transpose(1, 2, 0)

img = img[10:15, 15:20, :]

fig = plt.figure(figsize=(6, 6))

plt.imshow(img)

plt.show()

print('We are looking at image number', image_index, 'from training dataset')
print(img[:, :, 0])