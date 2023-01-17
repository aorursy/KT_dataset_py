!pip install -q efficientnet
import math, re, os



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import tensorflow as tf

import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

from sklearn import metrics

from sklearn.model_selection import train_test_split

import cv2

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
base_path = '/kaggle/input/alaska2-image-steganalysis/'

algorithm = ('Cover(Unaltered)', 'JMiPOD', 'UERD', 'JUNIWARD')

fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (11,11) )

np.random.seed(55)

for i,id in enumerate(np.random.randint(0,75001,4)):

    id = '{:05d}'.format(id)    

    cover_path = os.path.join(base_path, 'Cover', id + '.jpg')

    jmipod_path = os.path.join(base_path, 'JMiPOD', id + '.jpg')

    uerd_path = os.path.join(base_path, 'UERD', id + '.jpg')

    juniward_path = os.path.join(base_path, 'JUNIWARD', id + '.jpg')

    cover_img = plt.imread(cover_path)

    jmipod_img = plt.imread(jmipod_path)

    uerd_img = plt.imread(uerd_path)

    juniward_img = plt.imread(juniward_path)

    axes[i,0].imshow(cover_img)

    axes[i,1].imshow(jmipod_img)

    axes[i,2].imshow(uerd_img)

    axes[i,3].imshow(juniward_img)

    axes[i,0].set(ylabel=id+'.jpg')



for i,algo in enumerate(algorithm):

    axes[0,i].set(title=algo) 

for ax in axes.flat:

    ax.set(xticks=[], yticks=[])

plt.show()
cover_hist = {}

jmipod_hist = {}

uerd_hist = {}

juniward_hist = {}

color = ('b','g','r')

for i,col in enumerate(color):

    cover_hist[col] = cv2.calcHist([cover_img],[i],None,[256],[0,256])

    jmipod_hist[col] = cv2.calcHist([jmipod_img],[i],None,[256],[0,256])

    uerd_hist[col] = cv2.calcHist([uerd_img],[i],None,[256],[0,256])

    juniward_hist[col] = cv2.calcHist([juniward_img],[i],None,[256],[0,256])

    

fig_hist, axes_hist = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

for ax, hist, algo in zip(axes_hist.flat, [cover_hist, jmipod_hist, uerd_hist, juniward_hist], algorithm):

    ax.plot(hist['r'], color = 'r', label='r')

    ax.plot(hist['g'], color = 'g', label='g')

    ax.plot(hist['b'], color = 'b', label='b')

    ax.set(ylabel='# of pixels', xlabel='Pixel value(0-255)', title=algo)

    ax.legend()

fig_hist.subplots_adjust(wspace=0.4, hspace=0.3)

fig_hist.suptitle('Histogram of a sample (' + id + '.jpg)', fontsize=20)

    #     ax.xlim([0,256])

plt.show()
fig, ax = plt.subplots(figsize=(10,10))

ax.plot(cover_hist['r'][50:80], color = 'c', label=algorithm[0])

ax.plot(jmipod_hist['r'][50:80], color = 'm', label=algorithm[1])

ax.plot(uerd_hist['r'][50:80], color = 'y', label=algorithm[2])

ax.plot(juniward_hist['r'][50:80], color = 'g', label=algorithm[3])

ax.legend()

ax.set_ylabel('# of pixels', fontsize=15) 

ax.set_xlabel('Pixel value(50-80)', fontsize=15)

ax.xaxis.set(ticklabels=np.linspace(50,80,8, dtype=np.int))

ax.set_title('R-channel Histogram Compared (zoomed in)', fontsize=20)

plt.show()
fig, axes = plt.subplots(nrows=4, ncols=4, figsize = (11,11) )

np.random.seed(55)

def disp_diff_img(alt, ref, ax, chnl=0):

    diff = np.abs(alt.astype(np.int)-ref.astype(np.int)).astype(np.uint8)

    ax.imshow(diff[:,:,chnl], vmin=0, vmax=np.amax(diff[:,:,chnl]), cmap='hot')

for i,id in enumerate(np.random.randint(0,75001,4)):

    id = '{:05d}'.format(id)    

    cover_path = os.path.join(base_path, 'Cover', id + '.jpg')

    jmipod_path = os.path.join(base_path, 'JMiPOD', id + '.jpg')

    uerd_path = os.path.join(base_path, 'UERD', id + '.jpg')

    juniward_path = os.path.join(base_path, 'JUNIWARD', id + '.jpg')

    cover_img = plt.imread(cover_path)

    jmipod_img = plt.imread(jmipod_path)

    uerd_img = plt.imread(uerd_path)

    juniward_img = plt.imread(juniward_path)

    axes[i,0].imshow(cover_img)

    disp_diff_img(jmipod_img, cover_img, axes[i,1], 0)

    disp_diff_img(uerd_img, cover_img, axes[i,2], 0)

    disp_diff_img(juniward_img, cover_img, axes[i,3], 0)

    axes[i,0].set(ylabel=id+'.jpg')



for i,algo in enumerate(algorithm):

    axes[0,i].set(title=algo + 'diff') 

for ax in axes.flat:

    ax.set(xticks=[], yticks=[])

plt.show()
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

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

EPOCHS = 10

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
def append_path(pre):

    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))
sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')

train_filenames = np.array(os.listdir("/kaggle/input/alaska2-image-steganalysis/Cover/"))
np.random.seed(0)

positives = train_filenames.copy()

negatives = train_filenames.copy()

np.random.shuffle(positives)

np.random.shuffle(negatives)



jmipod = append_path('JMiPOD')(positives[:10000])

juniward = append_path('JUNIWARD')(positives[10000:20000])

uerd = append_path('UERD')(positives[20000:30000])



pos_paths = np.concatenate([jmipod, juniward, uerd])
test_paths = append_path('Test')(sub.Id.values)

neg_paths = append_path('Cover')(negatives[:30000])
train_paths = np.concatenate([pos_paths, neg_paths])

train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
train_paths, valid_paths, train_labels, valid_labels = train_test_split(

    train_paths, train_labels, test_size=0.15, random_state=2020)
def decode_image(filename, label=None, image_size=(512, 512)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

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
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .repeat()

    .shuffle(1024)

    .batch(BATCH_SIZE)

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

    .batch(BATCH_SIZE)

)
def build_lrfn(lr_start=0.00001, lr_max=0.000075, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn
with strategy.scope():

    model = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(512, 512, 3),

            weights='imagenet',

            include_top=False

        ),

        L.GlobalAveragePooling2D(),

        L.Dense(1, activation='sigmoid')

    ])

        

    model.compile(

        optimizer='adam',

        loss = 'binary_crossentropy',

        metrics=['accuracy']

    )

    model.summary()
# lrfn = build_lrfn()

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE



history = model.fit(

    train_dataset, 

    epochs=EPOCHS, 

#     callbacks=[lr_schedule],

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=valid_dataset

)
model.save("model.h5")
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    #ax.set_ylim(0.28,1.05)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
display_training_curves(

    history.history['loss'], 

    history.history['val_loss'], 

    'loss', 211)

display_training_curves(

    history.history['accuracy'], 

    history.history['val_accuracy'], 

    'accuracy', 212)
sub.Label = model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False)

sub.head()