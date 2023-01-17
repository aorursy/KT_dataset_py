%matplotlib inline



from kaggle_datasets import KaggleDatasets

import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pathlib import Path

import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE

DATASET_DIR = Path('/kaggle/input/flowers-tta')

PATH = Path('/kaggle/input/flower-classification-with-tpus')

SIZES = {s: f'{s}x{s}' for s in [192, 224, 331, 512]}

TFRECORD_DIR = KaggleDatasets().get_gcs_path(PATH.parts[-1])
classes_filename = DATASET_DIR/'classes.csv' 

CLASSES = tf.constant(pd.read_csv(classes_filename).values.squeeze(), tf.string)



def get_parse_fn(split):

    def parse_fn(example):

        features = {"image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

                    "id": tf.io.FixedLenFeature([], tf.string),

                    "class": tf.io.FixedLenFeature([], tf.int64)}

        

        if split == 'test':

            del features['class']

            

        example = tf.io.parse_single_example(example, features)

            

        example['image'] = tf.image.decode_jpeg(example['image'])

        example['label'] = tf.cast(example['class'], tf.int32)

        example['class'] = CLASSES[example['label']]

        

        return example



    return parse_fn



def get_ds(split, img_size=224, batch_size=128, shuffle=False):

    file_pat = f'{TFRECORD_DIR}/tfrecords-jpeg-{SIZES[img_size]}/{split}/*.tfrec'

    

    options = tf.data.Options()

    options.experimental_deterministic = not shuffle

    

    ds = (tf.data.Dataset.list_files(file_pat, shuffle=shuffle)

          .with_options(options)

          .interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)

          .map(get_parse_fn(split), num_parallel_calls=AUTO)

         )

    

    if shuffle:

        ds = ds.shuffle(2048)

            

    return ds.repeat().batch(batch_size).prefetch(AUTO)



def show_images(imgs, titles=None, hw=(3,3), rc=(4,4)):

    """Show list of images with optional list of titles."""

    h, w = hw

    r, c = rc

    fig=plt.figure(figsize=(w*c, h*r))

    gs1 = gridspec.GridSpec(r, c, fig, hspace=0.2, wspace=0.05)

    for i in range(r*c):

        img = imgs[i].squeeze()

        ax = fig.add_subplot(gs1[i])

        if titles != None:

            ax.set_title(titles[i], {'fontsize': 10})

        plt.imshow(img)

        plt.axis('off')

    plt.show()
ds_val = get_ds('val', shuffle=False)

ds_val_iter = ds_val.unbatch().batch(16).as_numpy_iterator()
b_val = next(ds_val_iter)

show_images(b_val['image'], b_val['class'].tolist(), hw=(2,2), rc=(2,8))
ds_trn = get_ds('train', shuffle=False)

ds_trn_iter = ds_trn.unbatch().batch(16).as_numpy_iterator()
b_trn = next(ds_trn_iter)

show_images(b_trn['image'], b_trn['class'].tolist(), hw=(2,2), rc=(2,8))
ds_sample = tf.data.experimental.sample_from_datasets([ds_trn.unbatch(), ds_val.unbatch()], [1., 1.])

ds_sample_iter = ds_sample.batch(16).as_numpy_iterator()
b_smp = next(ds_sample_iter)

show_images(b_smp['image'], b_smp['class'].tolist(), hw=(2,2), rc=(2,8))
choices = tf.data.Dataset.range(2).repeat()

ds_choose = tf.data.experimental.choose_from_datasets([ds_trn.unbatch(), ds_val.unbatch()], choices)

ds_choose_iter = ds_choose.batch(16).as_numpy_iterator()
b_ch = next(ds_choose_iter)

show_images(b_ch['image'], b_ch['class'].tolist(), hw=(2,2), rc=(2,8))
ds_zip = tf.data.Dataset.zip((ds_val.unbatch(), ds_trn.unbatch()))

ds_zip_iter = ds_zip.batch(8).as_numpy_iterator()
b_zip = next(ds_zip_iter)

show_images(b_zip[0]['image'], b_zip[0]['class'].tolist(), hw=(2,2), rc=(1,8))

show_images(b_zip[1]['image'], b_zip[1]['class'].tolist(), hw=(2,2), rc=(1,8))