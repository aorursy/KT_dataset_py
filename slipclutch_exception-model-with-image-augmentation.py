# load dependencies

import numpy as np 

import pandas as pd

import os

import shutil

import cv2

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

# load keras modules

import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam

from keras.regularizers import l2

from tensorflow.keras.applications import Xception

os.getcwd()

#os.chdir('/Users/Aron/Kaggle/plant_pathology')

local_dir = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7'

kaggle_dir = '/kaggle/input/plant-pathology-2020-fgvc7/'



sample_submission = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')

test = pd.read_csv(kaggle_dir + 'test.csv')

train = pd.read_csv(kaggle_dir + 'train.csv')

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

AUTO = tf.data.experimental.AUTOTUNE




try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()

    

IMG_SIZE = 300

def seed_everything(seed=0):

    np.random.seed(seed)

    tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'



seed = 2048

seed_everything(seed)

print("REPLICAS: ", strategy.num_replicas_in_sync)



def format_path(st):

    return GCS_DS_PATH + '/images/' + st + '.jpg'





sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



train_paths = train.image_id.apply(format_path).values

test_paths = test.image_id.apply(format_path).values

train_labels = train.loc[:, 'healthy':].values

SPLIT_VALIDATION =True

if SPLIT_VALIDATION:

    train_paths, valid_paths, train_labels, valid_labels =train_test_split(train_paths, train_labels, test_size=0.15, random_state=seed)



def decode_image(filename, label=None, IMG_SIZE=(IMG_SIZE, IMG_SIZE)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, IMG_SIZE)

    

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label
BATCH_SIZE = 32

train_dataset = (

tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

train_dataset_1 = (

tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(64)

    .prefetch(AUTO)

)

valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)



    
LR_START = 0.0001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.0001

LR_RAMPUP_EPOCHS = 4

LR_SUSTAIN_EPOCHS = 6

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
#The sample submission file, the training data and testing labels are read in.  ther eare 1821 training and testing images in the dataset.Next is to look at the distribution of the training set ategories.  



#There are 4 categories.  check to make sure there is a fatir representation of each of the 4 categories.



# since each image can only be represented in each column once, 

# the mean of the columns are the percentage each column is of the data.

print(train.sum())

pcts = train.mean()

pcts.plot(kind = 'bar')
from tensorflow.keras.applications import Xception

from keras.models import Model

from tensorflow import keras

with strategy.scope():

    Dense_net = Xception(

                    input_shape=(IMG_SIZE, IMG_SIZE, 3),

                    weights='imagenet',

                    include_top=False

                    )

    x = Dense_net.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(4, activation='softmax')(x)

    model =  keras.Model(inputs = Dense_net.input,outputs=x)

    model.compile(loss="categorical_crossentropy", optimizer= 'adam', metrics=["accuracy"])



# now create the data generator

datagen = ImageDataGenerator(

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)
#fit the model



model.fit(

    train_dataset,

    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,

    epochs=50,

    validation_data=valid_dataset if SPLIT_VALIDATION else None,)



predict= model.predict(test_dataset)

prediction = np.ndarray(shape = (test.shape[0],4), dtype = np.float32)

for row in range(test.shape[0]):

    for col in range(4):

        if predict[row][col] == max(predict[row]):

            prediction[row][col] = 1

        else:

            prediction[row][col] = 0

prediction = pd.DataFrame(prediction)

prediction.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

df = pd.concat([test.image_id, prediction], axis = 1)

df.to_csv('submission.csv', index = False)

from IPython.display import FileLink

FileLink(r'submission.csv')