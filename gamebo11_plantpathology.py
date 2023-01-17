# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        (os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt, cv2
import tensorflow as tf, re, math
path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'
all_imgs = os.listdir(path)
train_images = [train for train in all_imgs if 'Train' in train]
test_images = [test for test in all_imgs if 'Test' in test]
train_df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test_df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
def bytes_function(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def float_function(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def int64_function(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def serialize_example(feature0, feature1, feature2, feature3, feature4, feature5):
    feature = {
        'image' : bytes_function(feature0),
        'image_id': bytes_function(feature1),
        'healthy': int64_function(feature2),
        'multiple_diseases': int64_function(feature3),
        'rust': int64_function(feature4),
        'scab': int64_function(feature5)
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    return example.SerializeToString()
SIZE = 2020
CT = len(train_images)//SIZE+int(len(train_images)%SIZE!=0)

for j in range(CT):
    print(); print('Writing TFRecord %i of %i...'%(j,CT))
    CT2 = min(SIZE, len(train_images)-j*SIZE)
    with tf.io.TFRecordWriter('train%.2i-%i.tfrec'%(j, CT2)) as writer:
        for k in range(CT2):
            img = cv2.imread(path+train_images[SIZE*j+k])
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            name = train_images[SIZE*j+k].split('.')[0]
            row = train_df.loc[train_df.image_id == name]
            example = serialize_example( 
                img, str.encode(name),
                row.healthy.values[0],
                row.multiple_diseases.values[0],
                row.rust.values[0],
                row.scab.values[0]
            )
            writer.write(example)
            if k%100==0:
                print(k,', ', end='')
def test_serialize_example(feature0, feature1):
    feature = {
        'image' : bytes_function(feature0),
        'image_id': bytes_function(feature1)
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    return example.SerializeToString()
def bytes_features(value):
    if isinstance(value, tf.constant(0)):
        value = value.numpy()
    return tf.train.Features(bytes_list = (tf.train.Bytes_))
SIZE = 2020
CT = len(test_images)//SIZE+int(len(test_images)%SIZE!=0)

for j in range(CT):
    print('Writing {} of {} files to records...'.format(j, CT))
    CT2 = min(SIZE, len(test_images)-j*SIZE)
    with tf.io.TFRecordWriter('test%.2i-%i.tfrec'%(j, CT2)) as writer:
        for k in range(CT2):
            img = cv2.imread(path+test_images[SIZE*j+k])
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
            name = test_images[SIZE*j+k].split('.')[0]
            row = test_df.loc[test_df.image_id == name]
            example = test_serialize_example(
                img, str.encode(name))
            writer.write(example)
            if k%100==0:
                print(k,', ', end='')
raw_data = tf.data.TFRecordDataset('/kaggle/working/train00-1821.tfrec')
for raw_record in raw_data.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('TPU found:',tpu.master())
except ValueError:
    tpu = None
    print('No TPU was found')
    
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    
else:
    strategy = tf.distribute.get_strategy()

print('Number of replcias', strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE
batch_size = 16
image_shape = [224, 224]
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*image_shape, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),# tf.string means bytestring
        'healthy':tf.io.FixedLenFeature([], tf.int64),
        'multiple_diseases': tf.io.FixedLenFeature([], tf.int64),
        'rust': tf.io.FixedLenFeature([], tf.int64),
        'scab':tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    healthy = tf.cast(example['healthy'], tf.int32)
    multiple_diseases = tf.cast(example['multiple_diseases'], tf.int32)
    rust = tf.cast(example['rust'], tf.int32)
    scab = tf.cast(example['scab'], tf.int32)
    label = example['image_id']
    target = [healthy, multiple_diseases, rust, scab]
    return image, target # returns a dataset of (image, label) pairs

def read_unlabeled_data(eg):
    unlabeled_features = {
        'image' : tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string)
    }
    eg = tf.io.parse_single_example(eg, features=unlabeled_features)
    image = tf.image.decode_jpeg(eg['image'], channels=3)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.reshape(image, [*image_shape, 3])
    label = eg['image_id']
    return image, label

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_data)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, target):
    image = tf.image.random_flip_left_right(image, 101)
    image = tf.image.random_flip_up_down(image, 101)
    image = tf.image.random_brightness(image, .3, 101)
    image = tf.image.random_crop(image, [224, 224, 3], 42)
    return image, target

def get_training_dataset(do_aug = True):
    dataset = load_dataset('/kaggle/working/train00-1821.tfrec', labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    if do_aug: dataset = dataset.map(data_augment)
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_testing_dataset(ordered = True):
    dataset = load_dataset('/kaggle/working/test00-1821.tfrec', labeled = False, ordered = ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
train_dataset = get_training_dataset()
test_dataset = get_testing_dataset()
count
count = 0
for i in train_dataset:
    count+=1
print(count)
for image , target in cutmix_train.unbatch().take(1):
    plt.imshow(image)
    print(image.shape)
    print(target)
def cutmix(image, label, probability = 1.0):
    DIMS = 512
    CLASSES = 4
    
    imgs = []; labs = []
    for j in range(batch_size):
        P = tf.cast(tf.random.uniform([], 0, 1)<=probability, tf.int32)
        k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
        x = tf.cast(tf.random.uniform([], 0, DIMS), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIMS), tf.int32)
        b = tf.random.uniform([], 0, 1)
        WIDTH = tf.cast(DIMS * tf.math.sqrt(b), tf.int32)*P
        ya = tf.math.maximum(0, y-WIDTH//2)
        yb = tf.math.maximum(DIMS, y-WIDTH//2)
        xa = tf.math.maximum(0, x-WIDTH//2)
        xb = tf.math.maximum(DIMS, x-WIDTH//2)
        one = image[j, ya:yb, 0:xa, :]
        blank = np.zeros([*image_shape, 3])*tf.random.uniform([], 0, 255)
        two = blank[ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIMS, :]
        middle = tf.concat([one, two, three], axis = 1)
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIMS, :]], axis = 0)
        imgs.append(img)

#         a = tf.cast(WIDTH*WIDTH/DIMS/DIMS, tf.int32)

#         lab1 = label[j, ]
#         lab2 = label[k, ]

#         labs.append((1-a)*lab1+a*lab2)

    images = tf.reshape(tf.stack(imgs), (batch_size, 224, 224, 3))
#     labels = tf.reshape(tf.stack(labs), (batch_size, CLASSES))
    return images, label
cutmix_train = train_dataset.map(cutmix)
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.xception import Xception
with strategy.scope():
    pretrained_model = Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')
    model = tf.keras.models.Sequential([pretrained_model, tf.keras.layers.Dense(4, activation = 'softmax')])
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['AUC'])
    result = model.fit_generator(cutmix_train, epochs = 75, steps_per_epoch = 200)
test_images = test_dataset.map(lambda image, name:image)
preds = model.predict(test_images)
sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')
image_names = test_dataset.map(lambda image, label: label).unbatch()
names = next(iter(image_names.batch(1821))).numpy().astype('U')
sub_file = pd.concat([pd.DataFrame({'image_id':names}), pd.DataFrame(preds, columns = ['healthy','multiple_diseases', 'rust', 'scab'])], axis = 1)
sub_file.to_csv('Plants28.csv', index=False)
from numba import cuda

cuda.select_device()
cuda.close()
