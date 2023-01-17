!pip install -q efficientnet
%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



from tqdm import tqdm

from glob import glob

import gc



from sklearn.model_selection import train_test_split



import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import seaborn as sns

from IPython.display import display



plt.rcParams["figure.figsize"] = (12,8)

plt.rcParams['axes.titlesize'] = 16



from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import tensorflow.keras.layers as L



import efficientnet.tfkeras as efn



from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from tensorflow.keras.utils import to_categorical



import os

print(os.listdir('../input/alaska2-image-steganalysis'))
path = '/kaggle/input/alaska2-image-steganalysis/'

folders = ['JUNIWARD', 'JMiPOD', 'UERD', 'Cover']
sub = pd.read_csv('../input/alaska2-image-steganalysis/sample_submission.csv')

print(sub.shape)

sub.head()
plt.suptitle('Images from Cover', fontsize = 16)

for i, img in enumerate(os.listdir('../input/alaska2-image-steganalysis/Cover')[:12]):

    plt.subplot(3, 4, i + 1)

    img = mpimg.imread('../input/alaska2-image-steganalysis/Cover/' + img)

    plt.imshow(img)
plt.suptitle('Images from JMiPOD', fontsize = 16)

for i, img in enumerate(os.listdir('../input/alaska2-image-steganalysis/JMiPOD')[:12]):

    plt.subplot(3, 4, i + 1)

    img = mpimg.imread('../input/alaska2-image-steganalysis/JMiPOD/' + img)

    plt.imshow(img)
plt.suptitle('Images from JUNIWARD', fontsize = 16)

for i, img in enumerate(os.listdir('../input/alaska2-image-steganalysis/JUNIWARD')[:12]):

    plt.subplot(3, 4, i + 1)

    img = mpimg.imread('../input/alaska2-image-steganalysis/JUNIWARD/' + img)

    plt.imshow(img)
plt.suptitle('Images from UERD', fontsize = 16)

for i, img in enumerate(os.listdir('../input/alaska2-image-steganalysis/UERD')[:12]):

    plt.subplot(3, 4, i + 1)

    img = mpimg.imread('../input/alaska2-image-steganalysis/UERD/' + img)

    plt.imshow(img)
train_files = np.array(os.listdir(path + 'Cover/'))

print(len(train_files))

display(train_files[:5])
train_df = pd.DataFrame(columns = ['tag', 'file_path', 'target'])
np.random.seed(0)

np.random.shuffle(train_files)
start = 0

end = 10000

steg_paths = []

for tag in folders[:3]:

    for file in train_files[start: end]:

        full = tag + '/' + file

        steg_paths.append(full)

    start += 10000

    end += 10000

steg_paths[:5], len(steg_paths)
np.random.shuffle(train_files)

cover_paths = []

for file in train_files[:30000]:

    full = 'Cover' + '/' + file

    cover_paths.append(full)

cover_paths[-5:], len(cover_paths)
img_files = steg_paths + cover_paths

len(img_files)
train_df['file_path'] = img_files

train_df['tag'] = train_df['file_path'].apply(lambda x: x.split('/')[0])

flag = {'JUNIWARD': 1, 'JMiPOD': 1, 'UERD': 1, 'Cover': 0}

train_df['target'] = train_df['tag'].map(flag)

train_df = train_df.sample(frac = 1).reset_index(drop = True)

train_df
sns.countplot(train_df['tag'])
test_df = pd.DataFrame(columns = ['file_path', 'label'])



test_paths = []



for file in sub['Id']:

    full = 'Test' + '/' + file

    test_paths.append(full)



test_df['file_path'] = test_paths

test_df
print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)



BATCH_SIZE = 16 * strategy.num_replicas_in_sync

print(BATCH_SIZE)



GCS_DS_PATH = KaggleDatasets().get_gcs_path()

print(GCS_DS_PATH)
#Format path for TPU



def format_path(pt):

    return os.path.join(GCS_DS_PATH, pt)
train_paths = train_df['file_path'].apply(format_path).values

test_paths = test_df['file_path'].apply(format_path).values



train_targets = train_df['target'].values
train_paths[:5], train_targets[:5]
train_path, valid_path, train_label, valid_label = train_test_split(train_paths, train_targets, test_size = 0.2, random_state = 2019)

print(train_path.shape, train_label.shape, valid_path.shape, valid_label.shape)
def decode_image(filename, label = None, image_size = (512, 512)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels = 3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label = None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_path, train_label))

    .map(decode_image, num_parallel_calls = AUTO)

    .cache()

    .repeat()

    .shuffle(1024)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_path, valid_label))

    .map(decode_image, num_parallel_calls = AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(test_paths)

    .map(decode_image, num_parallel_calls = AUTO)

    .batch(BATCH_SIZE)

)
with strategy.scope():

    model = tf.keras.Sequential([

        efn.EfficientNetB7(

            input_shape = (512, 512, 3),

            weights = 'imagenet',

            include_top = False

        ),

        L.GlobalAveragePooling2D(),

        L.Dense(1, activation = 'sigmoid')

    ])

        

    model.compile(

        optimizer = 'adam',

        loss = 'binary_crossentropy',

        metrics = ['accuracy']

    )

    model.summary()
STEPS_PER_EPOCH = train_label.shape[0] // BATCH_SIZE



checkpoint = ModelCheckpoint('model_tpu.h5',monitor = 'val_loss', save_best_only = True, verbose = 1, period = 1)



reduceLR = ReduceLROnPlateau(monitor = 'val_loss', min_lr = 0.00001, patience = 3, mode = 'min', verbose = 1)
for l in model.layers:

    print(l)

    l.trainable = False

    

hist = model.fit(train_dataset, epochs = 3, steps_per_epoch = STEPS_PER_EPOCH, validation_data = valid_dataset,

                 callbacks = [checkpoint, reduceLR])



del hist

gc.collect()
EPOCHS = 10



history = model.fit(train_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_data = valid_dataset,

                 callbacks = [checkpoint, reduceLR])
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

    if subplot%10 == 1: # set up the subplots on the first call

        plt.subplots(figsize = (10,10), facecolor = '#F0F0F0')

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

    'Loss', 211)

display_training_curves(

    history.history['accuracy'], 

    history.history['val_accuracy'], 

    'Accuracy', 212)
preds = model.predict(test_dataset, verbose = 1)

sub.iloc[:, 1:] = preds

print(sub.shape)

display(sub.head())
sub.to_csv('./submission.csv', index = False)