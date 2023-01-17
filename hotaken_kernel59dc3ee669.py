!pip install efficientnet
import efficientnet.tfkeras as efn
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
# import efficientnet.tfkeras as efn
%pylab inline
import seaborn as sns
# from google.colab import files
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from kaggle_datasets import KaggleDatasets
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
    
def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 2048
seed_everything(seed)
print("REPLICAS: ", strategy.num_replicas_in_sync)

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
EPOCHS = 40
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values
train_labels = train.loc[:, 'healthy':].values
SPLIT_VALIDATION =True
if SPLIT_VALIDATION:
    train_paths, valid_paths, train_labels, valid_labels =train_test_split(train_paths, train_labels, test_size=0.15, random_state=seed)
image_size = 800
def decode_image(filename, label=None, image_size=(image_size, image_size)):
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
#     image = tf.image.random_brightness(image, 0.2)
    
    if label is None:
        return image
    else:
        return image, label
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

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
with strategy.scope():
    model = tf.keras.Sequential([
        InceptionResNetV2(
            input_shape=(image_size, image_size, 3),
            weights='imagenet',
            include_top=False
        ),
        L.GlobalMaxPooling2D(),
        L.Dense(4, activation='softmax')
#         SoftProbField()
    ])
        
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

history = model.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=[lr_callback],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None,
    verbose = 1,
)
%pwd
# probs1 = model.predict(test_dataset, verbose=1)
# probs2 = model3.predict(test_dataset, verbose=1)
# probs_avg = (probs1+probs2)/2
# sub.loc[:, 'healthy':] = probs_avg
# sub.to_csv('submission.csv', index=False)
# sub.head()
probs1 = model.predict(test_dataset)
probs1
sub.loc[:, 'healthy':] = probs1
sub.to_csv('submission.csv', index=False)
sub.head()
with strategy.scope():
    model2 = tf.keras.Sequential([
        efn.EfficientNetB7(
            input_shape=(image_size, image_size, 3),
            weights='noisy-student',
            include_top=False
        ),
         L.GlobalMaxPooling2D(),
        L.Dense(4, activation='softmax')
    ])

    model2.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model2.summary()
STEPS_PER_EPOCH = train_labels.shape[0] // 64

history = model2.fit(
    train_dataset_1, 
    epochs=EPOCHS, 
    callbacks=[lr_callback],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset if SPLIT_VALIDATION else None,
    verbose=1,
)
probs2 = model2.predict(test_dataset)
sub.loc[:, 'healthy':] = probs2
sub.to_csv('submission2.csv', index=False)
sub.head()
probs2

sub1 = pd.read_csv('/kaggle/input/submission/submission.csv')
sub11 = sub1.loc[:, 'healthy':].values
sub11
sub2 = pd.read_csv('/kaggle/working/submission2.csv')
sub22 = sub2.loc[:, 'healthy':].values
probs_avg = (sub11+sub22)/2
sub.loc[:, 'healthy':] = probs_avg
sub.to_csv('submissionavg.csv', index=False)
sub.head()