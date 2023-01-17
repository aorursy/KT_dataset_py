import cv2, math, os, sys

from tqdm import tqdm_notebook as tqdm

import zipfile

import io

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

import tensorflow as tf

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

warnings.filterwarnings("ignore")



PATH_IMAGE = '/kaggle/input/imagesgraph/'

train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')



train_df
!mkdir /kaggle/working/tfrecords
GCS_OUTPUT ='/kaggle/working/tfrecords/' # prefix for output file names

SHARDS = 128

TARGET_SIZE = [64, 64]
def resize_and_crop_image(image, label):

  w = tf.shape(image)[0]

  h = tf.shape(image)[1]

  tw = TARGET_SIZE[1]

  th = TARGET_SIZE[0]

  resize_crit = (w * th) / (h * tw)

  image = tf.cond(resize_crit < 1,

                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true

                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false

                 )

  nw = tf.shape(image)[0]

  nh = tf.shape(image)[1]

  image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)

  return image, label

  



def decode_jpeg_and_label(filename):

  bits = tf.io.read_file(filename)

  image = tf.image.decode_jpeg(bits)

  label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')

  label = label.values[-1]

  return image, label



filenames = tf.data.Dataset.list_files(PATH_IMAGE + '*.png', seed=35155) # This also shuffles the images

dataset1 = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)

dataset1 = dataset1.map(resize_and_crop_image, num_parallel_calls=AUTO)  
def display_9_images_from_dataset(dataset):

  plt.figure(figsize=(13,13))

  subplot=331

  for i, (image, label) in enumerate(dataset):

    img = image.numpy()

    img = cv2.resize(img,(64,64))

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(img.astype('float32'))

    plt.title(label.numpy().decode("utf-8"), fontsize=16)

    subplot += 1

    if i==2:

        break

  plt.tight_layout()

  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  plt.show()

display_9_images_from_dataset(dataset1)
def recompress_image(image, label):

  height = tf.shape(image)[0]

  width = tf.shape(image)[1]

  image = tf.cast(image, tf.uint8)

  image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

  return image, label, height, width

dataset3 = dataset1.map(recompress_image, num_parallel_calls=AUTO)

dataset3 = dataset3.batch(128) 
# FOR EXAMPLE I GET 10 BATCH, BUT IF YOU WANH CONVERT ALL IMAGES YOU NEED DELETE THIS CALL

dataset3 = dataset3.take(10)
head_root_hot_classes =  [x for x in range(168)]

head_vowel_hot_classes =  [x  for x in range(11)]

head_consonant_hot_classes = [x  for x in range(7)]

def get_hot_class(CLASSES, label):

    class_num = np.argmax(np.array(CLASSES)==label)

    return np.eye(len(CLASSES))[class_num]
def _bytestring_feature(list_of_bytestrings):

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))



def _int_feature(list_of_ints): # int64

  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))



def _float_feature(list_of_floats): # float32

  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

  



def to_tfrecord(tfrec_filewriter, img_bytes, label, height, width ,grapheme_root,vowel_diacritic,consonant_diacritic):  

  head_root_hot =  get_hot_class(head_root_hot_classes, grapheme_root)

  head_vowel_hot =  get_hot_class(head_vowel_hot_classes, vowel_diacritic)

  head_consonant_hot =  get_hot_class(head_consonant_hot_classes, consonant_diacritic)



  feature = {

      "image": _bytestring_feature([img_bytes]), 

      "grapheme_root": _int_feature([grapheme_root]),       

      "vowel_diacritic": _int_feature([vowel_diacritic]),       

      "consonant_diacritic": _int_feature([consonant_diacritic]),  



      "label":         _bytestring_feature([label]),         

      "size":          _int_feature([height, width]),     

      "head_root_hot": _float_feature(head_root_hot.tolist()),

      "head_vowel_hot": _float_feature(head_vowel_hot.tolist()),

      "head_consonant_hot": _float_feature(head_consonant_hot.tolist()),

  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

  

print("Writing TFRecords")

for shard, (image, label, height, width) in enumerate(dataset3):

  shard_size = image.numpy().shape[0]

  filename = GCS_OUTPUT + "{:02d}-{}.tfrec".format(shard, shard_size)

  

  with tf.io.TFRecordWriter(filename) as out_file:

    for i in range(shard_size):

        image_name = label.numpy()[i].decode("utf-8").split('.')[0]

        grapheme_root = train_df.loc[train_df.image_id == image_name]['grapheme_root'].values[0]

        vowel_diacritic = train_df.loc[train_df.image_id == image_name]['vowel_diacritic'].values[0]

        consonant_diacritic = train_df.loc[train_df.image_id == image_name]['consonant_diacritic'].values[0]

        example = to_tfrecord(out_file,

                            image.numpy()[i], # re-compressed image: already a byte string

                            label.numpy()[i],

                            height.numpy()[i],

                            width.numpy()[i],

                            grapheme_root,

                            vowel_diacritic,

                            consonant_diacritic)

        out_file.write(example.SerializeToString())

    print("Wrote file {} containing {} records".format(filename, shard_size))
def read_tfrecord(example):

    features = {

      "image": tf.io.FixedLenFeature([], tf.string), 

      "grapheme_root": tf.io.FixedLenFeature([], tf.int64),       

      "vowel_diacritic": tf.io.FixedLenFeature([], tf.int64),       

      "consonant_diacritic": tf.io.FixedLenFeature([], tf.int64),  



      "label":         tf.io.FixedLenFeature([], tf.string),         

      "size":          tf.io.FixedLenFeature([2], tf.int64),     

      "head_root_hot": tf.io.VarLenFeature(tf.float32),

      "head_vowel_hot": tf.io.VarLenFeature(tf.float32),

      "head_consonant_hot": tf.io.VarLenFeature(tf.float32),

    }



    example = tf.io.parse_single_example(example, features)



    image = tf.image.decode_image(example['image'], channels=3)

    image = tf.cast(image, tf.float32)/255.0 

    

    grapheme_root = example['grapheme_root']

    vowel_diacritic = example['vowel_diacritic']

    consonant_diacritic = example['consonant_diacritic']

     

    head_root_hot = tf.sparse.to_dense(example['head_root_hot'])

    head_vowel_hot = tf.sparse.to_dense(example['head_vowel_hot'])

    head_consonant_hot = tf.sparse.to_dense(example['head_consonant_hot'])

    

    head_root_hot = tf.reshape(head_root_hot, [168])

    head_vowel_hot = tf.reshape(head_vowel_hot, [11])

    head_consonant_hot = tf.reshape(head_consonant_hot, [7])

    

    label  = example['label']

    height = example['size'][0]

    width  = example['size'][1]

    return image,  {"head_root": head_root_hot, "head_vowel": head_vowel_hot, "head_consonant": head_consonant_hot}
import shutil

shutil.make_archive('gztfrecords', 'zip', '/kaggle/working/tfrecords')