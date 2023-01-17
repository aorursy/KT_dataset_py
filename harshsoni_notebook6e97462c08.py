import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from kaggle_datasets import KaggleDatasets
sns.set()
%matplotlib inline
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU: {}'.format(tpu.master()))
except ValueError:
    tpu = None
    
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print('REPLICAS :{}'.format(strategy.num_replicas_in_sync))
!ls -l '/kaggle/input'
GCS_DS_PATH = KaggleDatasets().get_gcs_path('nih-chest-xray-tfrecords')
!gsutil ls -l $GCS_DS_PATH
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMG_SIZE = [196, 196]
EPOCHS = 5
SHUFFLE = True
LR = 1e-2
AUTO = tf.data.experimental.AUTOTUNE

all_labels = [
    'Atelectasis', 
    'Cardiomegaly',
    'Consolidation', 
    'Edema', 
    'Effusion', 
    'Emphysema', 
    'Fibrosis', 
    'Hernia',
    'Infiltration', 
    'Mass', 
    'Nodule', 
    'Pleural_Thickening', 
    'Pneumonia',
    'Pneumothorax'
]
TRAIN_SIZE = 95466
VALID_SIZE = 11265
TEST_SIZE = 5389
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/nih/train/*.tfrecord')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/nih/validation/*.tfrecord')
TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/nih/test/*.tfrecord')
def decode_image(image, rescale=1.):
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) * rescale
    image = tf.image.resize(image, IMG_SIZE)
    
    return image

def preprocess_dataset(x, y, mean=None, std=None):
    if mean == None:
        mean = tf.math.reduce_mean(x)
    if std == None:
        std = tf.math.reduce_std(x)
    x = x - mean
    x = x / std
    
    y = tf.cast(y, tf.float32)
    
    return (x, y)

def process_dataset(dataset, REPEAT=True):
    dataset = dataset.cache()
    if SHUFFLE:
        dataset = dataset.shuffle(2048)
    if REPEAT:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    
    return dataset

def read_labelled_tfrecords(example):
    LABELLED_TFRECORD_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([14,], tf.float32)
    }
    
    example = tf.io.parse_single_example(example, LABELLED_TFRECORD_FORMAT)
    image = decode_image(example['image'])
    label = example['label']
    
    return image, label

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.map(read_labelled_tfrecords, num_parallel_calls=AUTO)
    
    return dataset

def _get_training_mean_std():
    train_ds = load_dataset(TRAINING_FILENAMES)
    
    sample_data = np.array(list(train_ds.take(1).as_numpy_iterator())[0][0])
    mean = tf.math.reduce_mean(sample_data, axis=tuple(range(2)))
    std = tf.math.reduce_std(sample_data, axis=tuple(range(2)))
    print('mean: {}\nstd: {}'.format(mean, std))
    
    return mean, std

def get_training_dataset():
    train_ds = load_dataset(TRAINING_FILENAMES)
    train_ds = train_ds.map(lambda x, y: preprocess_dataset(x, y), num_parallel_calls=AUTO)
    train_ds = process_dataset(train_ds)
    
    return train_ds

def get_validation_dataset(mean, std):
    valid_ds = load_dataset(VALIDATION_FILENAMES)
    valid_ds = valid_ds.map(lambda x, y: 
                            preprocess_dataset(x, y, mean=mean, std=std), num_parallel_calls=AUTO)
    valid_ds = process_dataset(valid_ds, REPEAT=False)
    
    return valid_ds

def get_test_dataset(mean, std):
    test_ds = load_dataset(TEST_FILENAMES)
    test_ds = valid_ds.map(lambda x, y: 
                           preprocess_dataset(x, y, mean=mean, std=std), num_parallel_calls=AUTO)
    test_ds = process_dataset(test_ds, REPEAT=False)
    
    return test_ds
mean, std = _get_training_mean_std()
def get_class_frequency(labels):
    pos_freq = np.mean(labels, axis=0)
    neg_freq = 1 - pos_freq
    
    return pos_freq, neg_freq

labels = pd.read_csv('../input/nih-chest-xray-tfrecords/nih/train_dataset.csv')[all_labels].values
pos_freq, neg_freq = get_class_frequency(labels)
neg_freq
pos_weight = neg_freq
neg_weight = pos_freq
def get_weighted_loss(pos_weight, neg_weight, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
#         loss = -tf.math.reduce_sum(tf.math.reduce_mean(pos_weight*tf.cast(y_true, tf.float32)*tf.math.log(y_pred+epsilon)
#                                                         +neg_weight*(1.-tf.cast(y_true, tf.float32))*tf.math.log(1-y_pred+epsilon), axis=0))
        loss = 0.0
    
        for i in range(len(pos_weight)):
            loss += -tf.math.reduce_mean(pos_weight[i]*y_true[:, i]*tf.math.log(y_pred[:, i] + epsilon) 
                                           + neg_weight[i]*(1-y_true[:, i])*tf.math.log(1-y_pred[:, i] + epsilon))
        
        return loss
    return weighted_loss
!mkdir ../working/tmp
!mkdir ../working/tmp/checkpoints
checkpoint_path = '../working/tmp/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=8e-1,
    patience=2,
    min_lr=1e-8,
    cooldown=5,
    verbose=1
)

earlystopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3
)
def create_model():
    dense_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet')

    x = dense_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions = tf.keras.layers.Dense(len(all_labels), activation='sigmoid')(x)
    model = tf.keras.Model(dense_model.input, predictions)
    
    return model
with strategy.scope():
    model = create_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=LR), 
    loss=get_weighted_loss(pos_weight, neg_weight),
    metrics=['binary_accuracy']
)

model.summary()
def train_model():
    history = model.fit(
    get_training_dataset(),
    epochs=EPOCHS,
    validation_data=get_validation_dataset(mean, std),
    steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
    validation_steps=VALID_SIZE // BATCH_SIZE,
    callbacks=[
         model_checkpoint_callback,
         reduce_lr_callback,
         earlystopping_callback
     ]
    )
    
train_model()
