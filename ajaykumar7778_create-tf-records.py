import numpy as np 

import pandas as pd 

import os

import cv2

import tensorflow as tf

from tqdm import tqdm

%matplotlib inline

from keras.preprocessing import image

import glob

import math, re, os

import sys



from sklearn.model_selection import train_test_split, StratifiedKFold
train_csv = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test_csv = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sample = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
GCS_DS_PATH = '../input/siim-isic-melanoma-classification'



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/train*')



TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*')
IMAGE_SIZE = [1024, 1024]

AUTO = tf.data.experimental.AUTOTUNE



BATCH_SIZE = 5



imSize = 1024
train_csv.head()
df = train_csv

df = df.drop(['target'], axis = 1)

df.head()
#X = train_csv['image_name']



X = df

y = train_csv['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_test = X_test.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)

X_train = X_train.reset_index(drop=True)

X_train
y_train
X_test.head()
a1 = X_test.loc[X_test['image_name'] == 'ISIC_7020578']['patient_id'].values

a1
#X_test['sex'] = X_test['sex'].fillna((X_test['sex'].mean()), inplace=True)
X_test[['sex', 'anatom_site_general_challenge', 'diagnosis']] = X_test[['sex', 'anatom_site_general_challenge', 'diagnosis']].replace(np.nan, '', regex=True)





X_train[['sex', 'anatom_site_general_challenge', 'diagnosis']] = X_train[['sex', 'anatom_site_general_challenge', 'diagnosis']].replace(np.nan, '', regex=True)
a1[0]
X_train['image_name'] == 'ISIC_8677254'
X_train['image_name'][0:10]
train_image_paths = []

train_image_labels = []



for i in range(len(X_train)):

    

    name = X_train['image_name'][i]

    path_temp = f'../input/siim-isic-melanoma-classification/jpeg/train/{name}.jpg'

    

    train_image_paths.append(path_temp)

    train_image_labels.append(y_train[i])

    

train_image_paths[0:10]
# The following functions can be used to convert a value to a type compatible

# with tf.Example.



def _bytes_feature(value):

    """Returns a bytes_list from a string / byte."""

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _int64_feature(value):

  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _floats_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
# Create a function to apply entire process to each element of dataset.

# process the two images into 'tf.Example' messages.

def image_example(image_string, label):

  """

  Creates a tf.Example message ready to be written to a file.

  """

  # Create a dictionary mapping the feature name to the tf.Example-compatible

  # data type.

  image_feature_description = {

      "image": _bytes_feature(image_string),

      "class": _int64_feature(label),

      }

  # Create a Features message using tf.train.Example.

  return tf.train.Example(features=tf.train.Features(feature=image_feature_description))
### Test code

def image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis):

    """

    Creates a tf.Example message ready to be written to a file.

    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible

    # data type.

    image_feature_description = {

      "image": _bytes_feature(image_string),

      "class": _int64_feature(label),

      #"patient_id": _bytes_feature(patient_id),

      "patient_id": _bytes_feature(value = patient_id.encode('utf-8')),

      "sex": _bytes_feature(value = sex.encode('utf-8')),

      "age_approx": _floats_feature(age_approx),

      "anatom_site_general_challenge": _bytes_feature(value = anatom_site_general_challenge.encode('utf-8')),

      "diagnosis": _bytes_feature(value = diagnosis.encode('utf-8')),

      }

  # Create a Features message using tf.train.Example.



    #print(image_feature_description)

    return tf.train.Example(features=tf.train.Features(feature=image_feature_description))



'''

# define a filename to store preprocessed image data:

record_file = '1.tfrecords'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[0:10]:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_train.loc[X_train['image_name'] == img_name]['patient_id'].values[0]

        sex = X_train.loc[X_train['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_train.loc[X_train['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_train.loc[X_train['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_train.loc[X_train['image_name'] == img_name]['diagnosis'].values[0]

        label = train_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())

        

        '''
#%env JOBLIB_TEMP_FOLDER=/tmp
#### Final code





def image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis):

    """

    Creates a tf.Example message ready to be written to a file.

    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible

    # data type.

    image_feature_description = {

      "image": _bytes_feature(image_string),

      "class": _int64_feature(label),

      #"patient_id": _bytes_feature(patient_id),

      "patient_id": _bytes_feature(value = patient_id.encode('utf-8')),

      "sex": _bytes_feature(value = sex.encode('utf-8')),

      "age_approx": _floats_feature(age_approx),

      "anatom_site_general_challenge": _bytes_feature(value = anatom_site_general_challenge.encode('utf-8')),

      "diagnosis": _bytes_feature(value = diagnosis.encode('utf-8')),

      }

  # Create a Features message using tf.train.Example.



    #print(image_feature_description)

    return tf.train.Example(features=tf.train.Features(feature=image_feature_description))





# define a filename to store preprocessed image data:

record_file = 'train-1.tfrec'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[0:5000]:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_train.loc[X_train['image_name'] == img_name]['patient_id'].values[0]

        sex = X_train.loc[X_train['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_train.loc[X_train['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_train.loc[X_train['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_train.loc[X_train['image_name'] == img_name]['diagnosis'].values[0]

        label = train_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())



        

'''        

record_file = 'train-2.tfrec'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[5000:10000]:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_train.loc[X_train['image_name'] == img_name]['patient_id'].values[0]

        sex = X_train.loc[X_train['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_train.loc[X_train['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_train.loc[X_train['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_train.loc[X_train['image_name'] == img_name]['diagnosis'].values[0]

        label = train_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())

        

record_file = 'train-3.tfrec'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[10000:15000]:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_train.loc[X_train['image_name'] == img_name]['patient_id'].values[0]

        sex = X_train.loc[X_train['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_train.loc[X_train['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_train.loc[X_train['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_train.loc[X_train['image_name'] == img_name]['diagnosis'].values[0]

        label = train_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())



record_file = 'train-4.tfrec'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[15000:20000]:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_train.loc[X_train['image_name'] == img_name]['patient_id'].values[0]

        sex = X_train.loc[X_train['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_train.loc[X_train['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_train.loc[X_train['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_train.loc[X_train['image_name'] == img_name]['diagnosis'].values[0]

        label = train_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())

        



record_file = 'train-5.tfrec'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[20000:25000]:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_train.loc[X_train['image_name'] == img_name]['patient_id'].values[0]

        sex = X_train.loc[X_train['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_train.loc[X_train['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_train.loc[X_train['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_train.loc[X_train['image_name'] == img_name]['diagnosis'].values[0]

        label = train_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())

        

        

record_file = 'train-6.tfrec'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[25000:]:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_train.loc[X_train['image_name'] == img_name]['patient_id'].values[0]

        sex = X_train.loc[X_train['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_train.loc[X_train['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_train.loc[X_train['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_train.loc[X_train['image_name'] == img_name]['diagnosis'].values[0]

        label = train_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())

        

        

        '''
valid_image_paths = []

valid_image_labels = []



for i in range(len(X_test)):

    

    name = X_test['image_name'][i]

    path_temp = f'../input/siim-isic-melanoma-classification/jpeg/train/{name}.jpg'

    

    valid_image_paths.append(path_temp)

    valid_image_labels.append(y_test[i])

    

valid_image_paths[0:10]
len(valid_image_paths)
### Validation data creation

'''

# define a filename to store preprocessed image data:

record_file = 'valid.tfrec'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in valid_image_paths:

        image_string = open(filename, 'rb').read()

        

        img_name = filename[54:-4]

        patient_id = X_test.loc[X_test['image_name'] == img_name]['patient_id'].values[0]

        sex = X_test.loc[X_test['image_name'] == img_name]['sex'].values[0]

        

        age_approx = X_test.loc[X_test['image_name'] == img_name]['age_approx'].values[0]

        anatom_site_general_challenge = X_test.loc[X_test['image_name'] == img_name]['anatom_site_general_challenge'].values[0]

        diagnosis = X_test.loc[X_test['image_name'] == img_name]['diagnosis'].values[0]

        label = valid_image_labels[index]

              

        index+=1

        

        ## storing all the features in the tf.Example message.

        tf_example = image_example2(image_string, label, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())

'''
#index
#anatom_site_general_challenge
#X_test.loc[X_test['image_name'] == img_name]['sex'].values
#img_name
#X_test.head()
#len(train_image_paths)
'''

# define a filename to store preprocessed image data:

record_file = 'images.tfrecords'

# Write the `tf.Example` observations to the file.

with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

    index = 0

    for filename in train_image_paths[0:10]:

        image_string = open(filename, 'rb').read()

        

        label = train_image_labels[index]

        index+=1

    ## storing all the features in the tf.Example message.

        tf_example = image_example(image_string, label)

    ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())

        

'''


'''

# Create a function to apply entire process to each element of dataset.

# process the two images into 'tf.Example' messages.

def image_example(image_string, label):

    """

    Creates a tf.Example message ready to be written to a file.

    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible

    # data type.

    image_feature_description = {

        "image": _bytes_feature(image_string),

        "class": _int64_feature(label),

        }

    # Create a Features message using tf.train.Example.

    print(image_feature_description)

    return tf.train.Example(features=tf.train.Features(feature=image_feature_description))

'''
'''

train_image_paths = []

train_image_labels = []



for i in range(len(X_train)):

    

    name = X_train['image_name'][i]

    path_temp = f'../input/siim-isic-melanoma-classification/jpeg/train/{name}.jpg'

    

    train_image_paths.append(path_temp)

    train_image_labels.append(y_train[i])

    

train_image_paths[0:10]'''
'''image_string = open(train_image_paths[0], 'rb').read()

image_string'''
## entire training



'''

# Write the `tf.Example` observations to the file.

cnt = 0

index = 0

record_file = f'train_{cnt}.tfrec'

for filename in train_image_paths[0:10]:

    

    image_string = open(filename, 'rb').read()

    cnt+=1

    if cnt%4000 == 0:

        record_file = f'train_{cnt}.tfrec'

    with tf.io.TFRecordWriter(record_file) as writer:

    #for filename, label in image_labels.items():

        

        

        label = train_image_labels[index]

        

        index+=1

        cnt+=1

        ## storing all the features in the tf.Example message.

        tf_example = image_example(image_string, label)

        ## write the example messages to a file named images.tfrecords

        writer.write(tf_example.SerializeToString())'''
'''!du -sh {record_file}'''
# to read TFRecord file use TFRecordDataset

raw_image_dataset = tf.data.TFRecordDataset(record_file)



# Create a dictionary describing the features.

image_feature_description = {

    "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

    "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }



# create a function to apply image feature description to each observation

def _parse_image_function(example_proto):

  # parse the input tf.Example proto using the dictionary above.

  return tf.io.parse_single_example(example_proto, image_feature_description)



# use map to apply this operation to each element of dataset

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset
# Use the .take method to only pull one example from the dataset.

for image_features in parsed_image_dataset.take(1):

    image = image_features['image'].numpy()

    #display.display(display.Image(data=image))

    classes = image_features['class'].numpy()

    print('The label of image is', classes)
image_features
'''

from PIL import Image

import io

image2 = Image.open(io.BytesIO(image))

'''
#image2
'''file = '/kaggle/working/images.tfrecords'

raw_image_dataset = tf.data.TFRecordDataset(file)'''
TRAINING_FILENAMES = ['/kaggle/working/valid.tfrec']
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    image = tf.image.resize(image, [imSize,imSize])

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



# data dump

'''

print("Training data shapes:")

for image, label in get_training_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)'''
#print("Training data label examples:", label.numpy())
#image_features['class'].numpy()
#display.Image(data=image)
'''train_image_paths = []

train_image_labels = []



for i in range(len(X_train)):

    

    name = X_train['image_name'][i]

    path_temp = f'../input/siim-isic-melanoma-classification/jpeg/train/{name}.jpg'

    

    train_image_paths.append(path_temp)

    train_image_labels.append(y_train[i])

 '''   
'''

def my_fn(img, label):

    a = tf.io.read_file(img)

    b = tf.image.decode_jpeg(a)

    #c = tf.image.resize_images(b, (192,192))

    d = tf.dtypes.cast(b, tf.uint8)

    e = tf.image.encode_jpeg(d)

    

    

    ## labels

    label = tf.cast(label, tf.int32)

    

    return e, label

    

ds = tf.data.Dataset.from_tensor_slices(train_image_paths[0:10], train_image_labels[0:10])



ds2 = ds.map(my_fn)



dds = ds2.map(tf.io.serialize_tensor)



tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')

tfrec.write(dds)

'''