import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image

import cv2

from tqdm.notebook import tqdm

import skimage.io

from skimage.transform import resize, rescale

import openslide
train_labels = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

train_labels.head()
data_dir = '../input/prostate-cancer-grade-assessment/train_images/'

mask_dir = '../input/prostate-cancer-grade-assessment/train_label_masks/'

mask_files = os.listdir(mask_dir)
img_id = train_labels.image_id[0]

path = data_dir + img_id + '.tiff'
%time biopsy = openslide.OpenSlide(path)

%time biopsy2 = skimage.io.MultiImage(path)
%timeit img = biopsy.get_thumbnail(size=(331, 331))

%timeit out = resize(biopsy2[-1], (331, 331))

%timeit out = cv2.resize(biopsy2[-1], (331, 331))

%timeit out = Image.fromarray(biopsy2[-1]).resize((331, 331))
out = cv2.resize(biopsy2[-1],(331,331))

%timeit Image.fromarray(out).save(img_id+'.png')

%timeit cv2.imwrite(img_id+'.png',out)
mask = skimage.io.MultiImage(mask_dir + mask_files[1])

img = skimage.io.MultiImage(data_dir + mask_files[1].replace("_mask", ""))
mask[-1].shape, img[-1].shape
save_dir = "/kaggle/train_images/"

os.makedirs(save_dir, exist_ok=True)
for img_id in tqdm(train_labels.image_id):

    load_path = data_dir + img_id + '.tiff'

    save_path = save_dir + img_id + '.png'

    

    biopsy = skimage.io.MultiImage(load_path)

    img = cv2.resize(biopsy[-1], (331, 331))

    cv2.imwrite(save_path, img)
save_mask_dir = '/kaggle/train_label_masks/'

os.makedirs(save_mask_dir, exist_ok=True)
for mask_file in tqdm(mask_files):

    load_path = mask_dir + mask_file

    save_path = save_mask_dir + mask_file.replace('.tiff', '.png')

    

    mask = skimage.io.MultiImage(load_path)

    img = cv2.resize(mask[-1], (331, 331))

    cv2.imwrite(save_path, img)
data_root = "/kaggle/train_images/"



tf_record_dir = os.path.join(data_root, "kaggle/tfrecord_data/")

tf_record_array_dir = os.path.join(data_root, "kaggle/tfrecord_array_data/")
import tensorflow as tf
def write_to_tfrecords(decoded_resolution=None):

    if decoded_resolution:

        record_dir = os.path.join(tf_record_array_dir, str(decoded_resolution))

    else:

        record_dir = tf_record_dir



    if os.path.exists(record_dir):

        return

    os.makedirs(record_dir, exist_ok=True)



    print("Converting images to TFRecords...")

    records_per_shard = 50



    shard_number = 0

    path_template = os.path.join(record_dir, "shard_{0:04d}.tfrecords")

    writer = tf.io.TFRecordWriter(path_template.format(shard_number))

    for i, (image_path, label) in enumerate(get_paths_and_labels()):

        if i and not (i % records_per_shard):

            shard_number += 1

            writer.close()

            writer = tf.io.TFRecordWriter(path_template.format(shard_number))



    with open(image_path, "rb") as f:

        image_bytes = f.read()



    if decoded_resolution:

        image = tf.io.decode_png(image_bytes)

        image = tf.cast(image, tf.float32)

        image = tf.image.resize(image, (decoded_resolution,) * 2)

        if image.shape[2] == 1:

            image = tf.tile(image, (1, 1, 3))

            image_bytes = tf.io.encode_jpeg(tf.cast(image, tf.uint8)).numpy()



    record_bytes = tf.train.Example(features=tf.train.Features(feature={

        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),

        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

    })).SerializeToString()



    writer.write(record_bytes)



    writer.close()

    print("TFRecord conversion complete.")





RECORD_PATTERN = os.path.join(tf_record_dir, "*.tfrecords")

RESIZED_RECORD_PATTERN = os.path.join(tf_record_array_dir, "{}", "*.tfrecords")

RECORD_SCHEMA = {

    "image": tf.io.FixedLenFeature([], dtype=tf.string),

    "label": tf.io.FixedLenFeature([1], dtype=tf.int64)

}



#write_to_tfrecords()



assert RESOLUTION[0] == RESOLUTION[1], "Resize is hard coded to square images."

write_to_tfrecords(RESOLUTION[0])
!tar -czf train_images.tar.gz ../train_images/*.png

!tar -czf train_label_masks.tar.gz ../train_label_masks/*.png