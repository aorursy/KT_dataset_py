# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import gc

import matplotlib.pyplot as plt







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

df.head()

df = df.sort_values("target", ascending = False)
TOTAL_SAMPLES = 1440
df_1 = df.iloc[:534]



df_1_val = df.iloc[534:584]



df_0 = df.iloc[584:TOTAL_SAMPLES]



df_0_val = df.iloc[TOTAL_SAMPLES: TOTAL_SAMPLES+100]

df_0 = df_0.sample(frac = 1)

df_0.head()
df_train = pd.concat([df_1,df_0])

df_train = df_train.sample(frac=1)



df_val = pd.concat([df_1_val, df_0_val])

df_val = df_val.sample(frac=1)

df_train
def preprocessing(dataset):



    dataset['sex'].fillna("no sex", inplace = True)

    dataset['age_approx'].fillna(0, inplace = True)

    dataset['anatom_site_general_challenge'].fillna("NA", inplace = True)

    dataset = dataset.replace(to_replace = ['male'], value = 0)

    dataset = dataset.replace(to_replace = ['female'], value = 1)

    dataset = dataset.replace(to_replace = ['no sex'], value = 2)

    dataset = dataset.replace(to_replace = ['torso'], value = 0)

    dataset = dataset.replace(to_replace = ['lower extremity'], value = 1)

    dataset = dataset.replace(to_replace = ['upper extremity'], value = 2)

    dataset = dataset.replace(to_replace = ['head/neck'], value = 3)

    dataset = dataset.replace(to_replace = ['NA'], value = 4)

    dataset = dataset.replace(to_replace = ['palms/soles'], value = 5)

    dataset = dataset.replace(to_replace = ['oral/genital'], value = 6)

    

    return dataset

df_train = preprocessing(df_train)

df_train
df_val = preprocessing(df_val)

df_val
df_test = preprocessing(pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv'))

df_test
def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    if (isinstance(value, type(tf.constant(0)))):

        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

    """Returns a float_list from a float / double."""

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _int64_feature(value):

    """Returns an int64_list from a bool / enum / int / uint."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, sex, age_approx,anatom_site_general_challenge, target, isTrain, image_name):



    if (isTrain):

        feature = {

            'sex': _int64_feature(sex),

            'age_approx': _float_feature(age_approx),

            'anatom_site_general_challenge': _int64_feature(anatom_site_general_challenge),

            'target': _int64_feature(target),

            'image_raw': _bytes_feature(image_string),

            }

    else:

        feature = {

            'image_name': _bytes_feature(image_name),

            'sex': _int64_feature(sex),

            'age_approx': _float_feature(age_approx),

            'anatom_site_general_challenge': _int64_feature(anatom_site_general_challenge),

            'image_raw': _bytes_feature(image_string),

            }

        

    return tf.train.Example(features=tf.train.Features(feature=feature))

IMAGE_HEIGHT = 300

IMAGE_WIDTH = 300
BATCH_SIZE = 16

SHUFFLE_SIZE = TOTAL_SAMPLES

STEPS_PER_EPOCH = int(TOTAL_SAMPLES/BATCH_SIZE)
record_file = '/kaggle/working/train.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:

    for row in df_train.iterrows():

#         image_string = open('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/' + row[1]['image_name'] + '.jpg', 'rb').read()

        im = cv2.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/' + row[1]['image_name'] + '.jpg')

        im_resize = cv2.resize(im, (IMAGE_HEIGHT, IMAGE_WIDTH))

        is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)

        image_string = im_buf_arr.tobytes()

        tf_example = image_example(image_string,row[1]['sex'], row[1]['age_approx'],row[1]['anatom_site_general_challenge'], row[1]['target'], isTrain = True,image_name = None)

        writer.write(tf_example.SerializeToString())

record_file = '/kaggle/working/validation.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:

    for row in df_val.iterrows():

#         image_string = open('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/' + row[1]['image_name'] + '.jpg', 'rb').read()

        im = cv2.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/' + row[1]['image_name'] + '.jpg')

        im_resize = cv2.resize(im, (IMAGE_HEIGHT, IMAGE_WIDTH))

        is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)

        image_string = im_buf_arr.tobytes()

        tf_example = image_example(image_string,row[1]['sex'], row[1]['age_approx'],row[1]['anatom_site_general_challenge'], row[1]['target'], isTrain = True,image_name = None)

        writer.write(tf_example.SerializeToString())
def _parse_function_train(proto):

    # define your tfrecord again. Remember that you saved your image as a string.

    keys_to_features = {'image_raw': tf.io.FixedLenFeature([], tf.string),

                        'target': tf.io.FixedLenFeature([], tf.int64),

                        'sex': tf.io.FixedLenFeature([], tf.int64),

                        'age_approx': tf.io.FixedLenFeature([], tf.float32),

                       'anatom_site_general_challenge': tf.io.FixedLenFeature([],tf.int64)}

    

    # Load one example

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    

    image_shape = tf.stack([IMAGE_HEIGHT,IMAGE_WIDTH,3])

#     target_shape = tf.stack([1])

    

    # Turn your saved image string into an array

    parsed_features['image_raw'] = tf.io.decode_jpeg(parsed_features['image_raw'], 3)#, fixed_length = 256*256*3)

    

#     parsed_features['image'] = tf.cast(parsed_features['image']/255, tf.float32)



    

    parsed_features['image_raw'] = tf.reshape(parsed_features['image_raw'], image_shape)

    

#     parsed_features['image_raw'] = tf.image.random_brightness(parsed_features['image_raw'],0.1)

    

#     parsed_features['image_raw'] = tf.image.random_contrast(parsed_features['image_raw'],0.1,0.55)

    

    parsed_features['image_raw'] = tf.image.random_flip_left_right(parsed_features['image_raw'])

    

    parsed_features['image_raw'] = tf.image.random_flip_up_down(parsed_features['image_raw'])

    

#     parsed_features['image_raw'] = tf.image.random_saturation(parsed_features['image_raw'], 5, 10, seed=None)



#     parsed_features['image_raw'] = tf.image.adjust_saturation(parsed_features['image_raw'],10)

    

#     parsed_features['image_raw'] = tf.image.random_saturation(parsed_features['image_raw'], 1, 5, seed=None)



    parsed_features['image_raw'] = tf.image.adjust_saturation(parsed_features['image_raw'],1)

    

#     parsed_features['image_raw'] = tf.image.adjust_contrast(parsed_features['image_raw'],2)



    

#     parsed_features['image_raw'] = tf.image.adjust_jpeg_quality(parsed_features['image_raw'],75)

    

    

#     parsed_features["target"] = tf.reshape(parsed_features['target'],target_shape)

    

    return parsed_features['image_raw'], parsed_features['sex'], parsed_features['age_approx'], parsed_features['anatom_site_general_challenge'],parsed_features['target']
def dataset_fetch (filenames, isTrain):

    

    dataset = tf.data.TFRecordDataset(filenames)

    

    if(isTrain == True):

        dataset = dataset.repeat()



    dataset = dataset.map(_parse_function_train)



#     else:

#         dataset = dataset.map(_parse_function_test)

        

#     dataset = dataset.shuffle(SHUFFLE_SIZE)



    dataset = dataset.batch(BATCH_SIZE)

        

#     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            

    return dataset
# from PIL import Image

training_dataset = dataset_fetch('./train.tfrecords',True)

# training_dataset = training_dataset.repeat(-1)



for element in training_dataset:

    break
print(element[4]) #target
image_index=4



print(element[4][image_index])

plt.imshow(element[0][image_index])
plt.imshow(element[0][image_index][:,:,1],cmap = 'gray', clim = (75,250))
plt.hist(np.array(element[0][image_index][:,:,2]))