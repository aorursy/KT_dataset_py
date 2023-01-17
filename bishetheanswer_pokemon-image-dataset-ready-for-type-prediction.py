import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

import pandas as pd



import os

import glob

import collections

import shutil

from PIL import Image

import matplotlib.pyplot as plt





# Set the seed

seed = 27912

np.random.seed(seed)

data = pd.read_csv("../input/pokemon-images-and-types/pokemon.csv")
data.sample(5)
images_dir = "../input/pokemon-images-and-types/images/images/"
data.info
os.chdir(images_dir)

cnt = collections.Counter()

for filename in glob.glob("*"):

    name, ext = os.path.splitext(filename)

    cnt[ext] += 1

print(cnt)
types = data.Type1.unique()

types
os.mkdir("newData")



for t in types:

    os.mkdir("newData/{}".format(t))
for t in types:

    aux_type = data[data.Type1.eq(t)]

    for pokemon in aux_type.Name:

        for filename in os.listdir(images_dir):

            original_path = "{}{}".format(images_dir, filename)

            # pokemon name with extension

            extension = os.path.basename(original_path)

            # directory without extension

            poke_dir = os.path.splitext(original_path)[0]

            # only pokemon name

            poke_name = os.path.basename(poke_dir)

            if(pokemon == poke_name):

                target_path = "newData/{}/{}".format(t, extension)

                shutil.copyfile(original_path, target_path)

fill_color = (255, 255, 255)

new_images_dir = "newData/"



for t in types:

    for filename in os.listdir(new_images_dir):

        type_dir = "{}{}/".format(new_images_dir, filename)

        for pokemon in os.listdir(type_dir):

            full_path = "{}/{}".format(type_dir, pokemon)

            file_dir, file_extension = os.path.splitext(full_path)

            if file_extension == ".png":

                im = Image.open(full_path)

                im = im.convert("RGBA")

                if im.mode in ('RGBA', 'LA'):

                    bg = Image.new(im.mode[:-1], im.size, fill_color)

                    bg.paste(im, im.split()[-1])  # omit transparency

                    bg.save("{}.jpg".format(file_dir))

                    os.remove(full_path)

dir_name = "newData/"

output_filename = "newData"



shutil.make_archive(output_filename, 'zip', dir_name)

batch_size = 16

img_height = 120

img_width = 120
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

  new_images_dir,

  validation_split=0.2,

  subset="training",

  seed=seed,

  image_size=(img_height, img_width),

  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

  new_images_dir,

  validation_split=0.2,

  subset="validation",

  seed=seed,

  image_size=(img_height, img_width),

  batch_size=batch_size)
class_names = train_ds.class_names

print(class_names)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")