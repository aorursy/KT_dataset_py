from kaggle_datasets import KaggleDatasets



import tensorflow as tf



# работа с изображениями

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

%matplotlib inline 
# Устанавливаем seed рандомизатора

import os

import numpy as np

from tensorflow.random import set_seed

def seed_everything(seed):

    np.random.seed(seed)

    set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 42

seed_everything(seed)
# Подсасываем датасет

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')

print(f'GCS_DS_PATH = {GCS_DS_PATH}\n')

IMAGE_SIZE = (512, 512)



HEIGHT = IMAGE_SIZE[0]

WIDTH = IMAGE_SIZE[1]

CHANNELS = 3

BATCH_SIZE = 32



GCS_PATH_SELECT = {

    (192, 192) : GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    (224, 224): GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    (331, 331): GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    (512, 512): GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

# Дергаем путь к картинкам в нужном разрешении

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE]

print(GCS_PATH)



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')
AUTO = tf.data.experimental.AUTOTUNE



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [HEIGHT, WIDTH, 3])

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label



def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False 



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    return dataset



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset

# Функции аугментации изображения

def random_brightness(image, label):

    image = tf.image.random_brightness(image, max_delta=0.5, seed=seed)

    return image, label



def random_contrast(image, label):

    image = image = tf.image.random_contrast(image, lower=.2, upper=3, seed=seed)

    return image, label



def random_saturation(image, label):

    image = tf.image.random_saturation(image, lower=0, upper=2, seed=seed)

    return image, label



def random_crop(image, label):

    image = tf.image.random_crop(image, size=[int(HEIGHT*.8), int(WIDTH*.8), CHANNELS], seed=seed)

    return image, label



def random_flip(image, label):

    if all(np.random.randint(2, size=1)):

        image = tf.image.random_flip_up_down(image, seed=seed)

    if all(np.random.randint(2, size=1)):

        tf.image.random_flip_left_right(image, seed=seed)

    return image, label



def random_hue(image, label):

    image = tf.image.random_hue(image, max_delta=0.5, seed=seed)

    return image, label



def random_quality(image, label):

    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=1,max_jpeg_quality=100, seed=seed)

    return image, label
def make_plot(row, col, augmented_element, name):

    for (img,label) in augmented_element:

        plt.figure(figsize=(15,int(15*row/col)))

        

        for j in range(row*col):

            plt.subplot(row,col,j+1)

            plt.axis('off')

            plt.imshow(img[j,])

        plt.suptitle(name)

        plt.show()

        break
def augment_with(augment_function):

    all_elements = get_training_dataset().unbatch()

    one_element = tf.data.Dataset.from_tensors( next(iter(all_elements)) )

    augmented_element = one_element.repeat().map(augment_function).batch(row*col)



    make_plot(row, col,augmented_element,  augment_function.__name__)

    
row = 3; col = 4;

augment_with(random_brightness)



augment_with(random_contrast)

augment_with(random_saturation)
augment_with(random_crop)
augment_with(random_flip)
augment_with(random_hue)
augment_with(random_quality)
# Использование ImageDataGenerator

all_elements = get_training_dataset().unbatch()

one_element = next(iter(all_elements))

samples = np.expand_dims(one_element[0], 0)



datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True

)

datagen.fit(samples)

it = datagen.flow(samples, batch_size=1)

# Эмулируем эпохи

row = 3; col = 4;



plt.figure(figsize=(15,int(15*row/col)))

for i in range(row*col):

    plt.subplot(row,col,i+1)

    plt.axis('off')

    batch = it.next()

    image = batch[0].astype('uint8')

    plt.imshow(image)

plt.show()