import os

import re

import math



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



import tensorflow as tf

import cv2 as cv

import seaborn as sns
BASE_PATH = '../input/siim-isic-melanoma-classification/'
TRAIN_IMAGE_PATH = os.path.join(BASE_PATH, 'jpeg', 'train')

TEST_IMAGE_PATH  = os.path.join(BASE_PATH, 'jpeg', 'test')



TRAIN_IMAGE_LIST = os.listdir(TRAIN_IMAGE_PATH)

TEST_IMAGE_LIST  = os.listdir(TEST_IMAGE_PATH)



print('There are %i train images and %i test images' % (len(TRAIN_IMAGE_LIST), len(TEST_IMAGE_LIST)))
df_train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))

df_train.rename({'image_id': 'image_name'}, axis=1, inplace=True)

df_train.head()
df_test = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

df_test.head()
df_combined = pd.concat([df_train[df_test.columns], df_test[df_test.columns]], ignore_index=True, axis=0).reset_index(drop=True) # Combine test and train to encode together
# Label Encode all the strings

labels_categorical = ['patient_id','sex','anatom_site_general_challenge'] 

for label in labels_categorical:

    df_combined[label], mp = df_combined[label].factorize()

    print(mp)



# Mean Encode the Age NaN valies

print('Imputing Age NaN count =', df_combined.age_approx.isnull().sum())

df_combined.age_approx.fillna(df_combined.age_approx.mean(), inplace=True)

df_combined['age_approx'] = df_combined.age_approx.astype('int')
# Rewrite encoded data to original dataframes

labels_categorical = labels_categorical + ['age_approx']

df_train[labels_categorical] = df_combined.loc[ : df_train.shape[0] - 1, labels_categorical].values

df_test[labels_categorical]  = df_combined.loc[df_train.shape[0] : , labels_categorical].values
# Label encode the site of the image

df_train.diagnosis, mp = df_train.diagnosis.factorize()

print(mp)
df_train = df_train.drop(['benign_malignant'], axis=1)

df_train.head()
def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _int64_feature(value):

    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def hair_remove(image):

    # convert image to grayScale

    grayScale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # kernel for morphologyEx

    kernel = cv.getStructuringElement(1,(17,17))

    # apply MORPH_BLACKHAT to grayScale image

    blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat

    _,threshold = cv.threshold(blackhat,10,255,cv.THRESH_BINARY)

    # inpaint with original image and threshold image

    final_image = cv.inpaint(image,threshold,1,cv.INPAINT_TELEA)

    

    return final_image
def serialize_train(image, image_name, patient_id, sex, age, site, diagnosis, target):

    feature = {

      'image': _bytes_feature(image),

      'image_name': _bytes_feature(image_name),

      'patient_id': _int64_feature(patient_id),

      'sex': _int64_feature(sex),

      'age_approx': _int64_feature(age),

      'anatom_site_general_challenge': _int64_feature(site),

      'diagnosis': _int64_feature(diagnosis),

      'target': _int64_feature(target)

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
BATCH_SIZE = 2000

BATCH_COUNT = len(TRAIN_IMAGE_LIST) // BATCH_SIZE + int(len(TRAIN_IMAGE_LIST) % BATCH_SIZE != 0)



for j in range(BATCH_COUNT):

    print('\nWriting TFRecord %i of %i...' % (j, BATCH_COUNT))

    CURRENT_BATCH_SIZE = min(BATCH_SIZE, len(TRAIN_IMAGE_LIST) - j * BATCH_SIZE)

    with tf.io.TFRecordWriter('train%.2i-%i.tfrec'%(j, CURRENT_BATCH_SIZE)) as writer:

        for k in range(CURRENT_BATCH_SIZE):

            img = cv.imread(os.path.join(TRAIN_IMAGE_PATH, TRAIN_IMAGE_LIST[BATCH_SIZE * j + k]))

            img = cv.resize(img, (256, 256), interpolation = cv.INTER_AREA)

            img = hair_remove(img)

            img = cv.imencode('.jpg', img, (cv.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

            name = TRAIN_IMAGE_LIST[BATCH_SIZE * j + k].split('.')[0]

            row = df_train.loc[df_train.image_name == name]

            example = serialize_train(

                img, 

                str.encode(name),

                row.patient_id.values[0],

                row.sex.values[0],

                row.age_approx.values[0],

                row.anatom_site_general_challenge.values[0],

                row.diagnosis.values[0],

                row.target.values[0]

            )

            writer.write(example)

            if (k % 100 == 0): print(k // 100, end=' ')
os.listdir('.')
def serialize_test(image, image_name, patient_id, sex, age, site): 

    feature = {

      'image': _bytes_feature(image),

      'image_name': _bytes_feature(image_name),

      'patient_id': _int64_feature(patient_id),

      'sex': _int64_feature(sex),

      'age_approx': _int64_feature(age),

      'anatom_site_general_challenge': _int64_feature(site),

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
BATCH_SIZE = 500

BATCH_COUNT = len(TEST_IMAGE_LIST) // BATCH_SIZE + int(len(TEST_IMAGE_LIST) % BATCH_SIZE != 0)



for j in range(BATCH_COUNT):

    print(); print('Writing TFRecord %i of %i...' % (j, BATCH_COUNT))

    CURRENT_BATCH_SIZE = min(BATCH_SIZE,len(TEST_IMAGE_LIST) - j * BATCH_SIZE)

    with tf.io.TFRecordWriter('test%.2i-%i.tfrec' % (j, CURRENT_BATCH_SIZE)) as writer:

        for k in range(CURRENT_BATCH_SIZE):

            img = cv.imread(os.path.join(TEST_IMAGE_PATH, TEST_IMAGE_LIST[BATCH_SIZE * j + k]))

            img = cv.resize(img, (256, 256), interpolation = cv.INTER_AREA)

            img = hair_remove(img)

            img = cv.imencode('.jpg', img, (cv.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

            name = TEST_IMAGE_LIST[BATCH_SIZE * j + k].split('.')[0]

            row = df_test.loc[df_test.image_name == name]

            example = serialize_test(

                img,

                str.encode(name),

                row.patient_id.values[0],

                row.sex.values[0],

                row.age_approx.values[0],

                row.anatom_site_general_challenge.values[0]

            )

            writer.write(example)

            if (k % 100 == 0): print(k // 100, end=' ')
os.listdir('.')
AUTO = tf.data.experimental.AUTOTUNE

cfg = dict(

    read_size = 512,

    crop_size = 500, 

    net_size  = 448, 

)



def read_labeled_tfrecord(example, return_image_name):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),

        'sex'                          : tf.io.FixedLenFeature([], tf.int64),

        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),

        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),

        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),

        'target'                       : tf.io.FixedLenFeature([], tf.int64)

    }           

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['sex'], example['age_approx'], example['anatom_site_general_challenge'], example['image_name'] if return_image_name else example['target']





def read_unlabeled_tfrecord(example, return_image_name):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),

        'sex'                          : tf.io.FixedLenFeature([], tf.int64),

        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),

        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64)

    }

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['sex'], example['age_approx'], example['anatom_site_general_challenge'], example['image_name'] if return_image_name else 0



 

def prepare_image(img, augment=True):    

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, [cfg['read_size'], cfg['read_size']])

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.image.central_crop(img, cfg['crop_size'] / cfg['read_size'])                               

    img = tf.image.resize(img, [cfg['net_size'], cfg['net_size']])

    img = tf.reshape(img, [cfg['net_size'], cfg['net_size'], 3])

    return img



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)





def get_dataset(files, labeled=True, return_image_names=True):

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO).cache()

    

    if labeled:

        ds = ds.map(lambda example: read_labeled_tfrecord(example, return_image_names), num_parallel_calls=AUTO)

    else:

        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), num_parallel_calls=AUTO)

    

    ds = ds.map(lambda img, sex, age, site, label: tuple([tuple([prepare_image(img), sex, age, site]), label]), 

            num_parallel_calls=AUTO)

    ds = ds.batch(32)

    ds = ds.prefetch(AUTO)

    return ds
TRAIN_RECORDS = [ file for file in os.listdir('.') if 'train' in file ]

TEST_RECORDS = [ file for file in os.listdir('.') if 'test' in file ]
fig, ax = plt.subplots(5, 2, figsize=(10, 25))



ds = get_dataset(TRAIN_RECORDS, labeled=True).unbatch().take(5)

for idx, item in enumerate(ds):

    ax[idx][0].imshow(item[0][0])

    original = plt.imread(os.path.join(BASE_PATH, 'jpeg', 'train', item[1].numpy().decode("utf-8") + '.jpg'))

    ax[idx][1].imshow(original)

    print('Sex: %s, Age: %s, Site: %s'%(item[0][1], item[0][2], item[0][3]))
fig, ax = plt.subplots(5, 2, figsize=(10, 25))



ds = get_dataset(TEST_RECORDS, labeled=False).unbatch().take(5)

for idx, item in enumerate(ds):

    ax[idx][0].imshow(item[0][0])

    original = plt.imread(os.path.join(BASE_PATH, 'jpeg', 'test', item[1].numpy().decode("utf-8") + '.jpg'))

    ax[idx][1].imshow(original)

    print('Sex: %s, Age: %s, Site: %s'%(item[0][1], item[0][2], item[0][3]))
ds = get_dataset(TEST_RECORDS, labeled=False, return_image_names=True)

image_names = np.array([img_name.numpy().decode("utf-8") for img, img_name in iter(ds.unbatch())])
print(image_names.shape)

print(np.unique(image_names).shape)
# !tar czf test.tar.gz test*.tfrec
# !tar czf train.tar.gz train*.tfrec