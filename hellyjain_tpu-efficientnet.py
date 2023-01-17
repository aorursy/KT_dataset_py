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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#install jpegio
# ! git clone https://github.com/hellyjain/jpegio
# Once downloaded install the package
# !pip install jpegio/.
!pip install -q efficientnet
import os
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import jpegio as jpio
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import gc
#compare cover with corresponding stegnograph picture


def compare_steg(image):    
    fig, ax = plt.subplots(nrows= 3, ncols = 3, figsize = (22,17))
    steg = ['JMiPOD', 'JUNIWARD', 'UERD']
    cover = cv2.imread('/kaggle/input/alaska2-image-steganalysis/Cover/'+image)
    for r in range(3):
        coded = cv2.imread('/kaggle/input/alaska2-image-steganalysis/'+str(steg[r])+'/'+image)
        for c in range(3):
            ax[r][1].set_title(steg[r])
            ax[r][0].imshow(cover)
            ax[r][0].axis('off')
            ax[r][1].imshow(coded)
            ax[r][1].axis('off')
            ax[r][2].imshow(cover - coded)
            ax[r][2].axis('off')

            
#Lets do pixel by pixel comparision of raw matrix

def pixel_summary(image):
    cover = cv2.imread('/kaggle/input/alaska2-image-steganalysis/Cover/'+image)
    JMiPOD = cv2.imread('/kaggle/input/alaska2-image-steganalysis/JMiPOD/'+image)
    JUNIWARD = cv2.imread('/kaggle/input/alaska2-image-steganalysis/JUNIWARD/'+image)
    UERD = cv2.imread('/kaggle/input/alaska2-image-steganalysis/UERD/'+image)
    u_cover = np.unique(cover.reshape(512*512,3), axis =0).shape
    u_JMiPOD = np.unique(JMiPOD.reshape(512*512,3), axis =0).shape
#     d_JMiPOD = np.unique((cover - JMiPOD).reshape(512*512,3), axis =0).shape
    u_JUNIWARD = np.unique(JUNIWARD.reshape(512*512,3), axis =0).shape
#     d_JUNIWARD = np.unique((cover - JUNIWARD).reshape(512*512,3), axis =0).shape
    u_UERD = np.unique(UERD.reshape(512*512,3), axis =0).shape
#     d_UERD = np.unique((cover - UERD).reshape(512*512,3), axis =0).shape
    print('Total unique colors in original image : {}'.format(u_cover[0]))
    print('Total unique colors in JMiPOD image : {}'.format(u_JMiPOD[0]))
#     print('Change in colors comparing Original vs JMiPOD image : {}'.format((d_JMiPOD)[0]))
    print('Total unique colors in JUNIWARD image : {}'.format(u_JUNIWARD[0]))
#     print('Change in colors comparing Original vs JUNIWARD image : {}'.format((d_JUNIWARD)[0]))
    print('Total unique colors in UERD image : {}'.format(u_UERD[0]))
#     print('Change in colors comparing Original vs UERD image : {}'.format((d_UERD)[0]))
compare_steg('00001.jpg')
pixel_summary('00001.jpg')
compare_steg('00501.jpg')
pixel_summary('00501.jpg')
compare_steg('01011.jpg')
pixel_summary('01011.jpg')
compare_steg('62001.jpg')
pixel_summary('62001.jpg')
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path('alaska2-image-steganalysis')

# Configuration
# EPOCHS = 10
BATCH_SIZE = 16 * strategy.num_replicas_in_sync * 2
GCS_DS_PATH
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
    
# def decode_image_YCbCr(filename, label=None, image_size=(512, 512)):
# #     bits = tf.io.read_file(filename)
#     image = JPEGdecompressYCbCr_v3(filename)
#     image = tf.cast(image, tf.float32) / 256.0
#     image = tf.image.resize(image, image_size)
    
#     if label is None:
#         return image
#     else:
#         return image, label

# def data_augment(image, label=None):
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
    
#     if label is None:
#         return image
#     else:
#         return image, label
with strategy.scope():
    model = tf.keras.Sequential([efn.EfficientNetB5(input_shape=(512, 512, 3),weights='imagenet',include_top=False)
                             ,L.GlobalAveragePooling2D()
                            ,L.Dense(10, activation='softmax')])
    model.compile(optimizer='adamax', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
train = pd.read_csv('/kaggle/input/training/input_batch.csv')
train['imagepath'] = train.imagepath.str.replace('gs://kds-cd794677eb4416f86e3e23278e2d1f1e37783bc29e7982b08d469689',
                        GCS_DS_PATH, regex = True)
train.head()
for i in tqdm(range(5,6)):
    train_paths = train[train['batch'] == i]['imagepath']
    train_lables = train[train['batch'] == i]['label']
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_lables, test_size=0.20, random_state=619)
    train_dataset = (tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).map(decode_image, num_parallel_calls=AUTO).cache().repeat().shuffle(1024).batch(BATCH_SIZE).prefetch(AUTO))
    valid_dataset = (tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels)).map(decode_image, num_parallel_calls=AUTO).batch(BATCH_SIZE).cache().prefetch(AUTO))
    STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
    history = model.fit(train_dataset, epochs=10, batch_size = BATCH_SIZE, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_dataset)
    gc.collect()
#Read the test images

def append_path(pre):
    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))

submission = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')
test_paths = append_path('Test')(submission.Id.values)
#create submission file

test_dataset = (tf.data.Dataset.from_tensor_slices(test_paths).map(decode_image, num_parallel_calls=AUTO).batch(BATCH_SIZE))
lab = model.predict(test_dataset, verbose=1)
submission.Label = 1-lab[:,[0]]
submission.to_csv('submission_ALASKA.csv', index=False)
#Save model weights and use them while training other batches of images

model.save_weights("model_b5.h5")
# model.load_weights("model_b5.h5")