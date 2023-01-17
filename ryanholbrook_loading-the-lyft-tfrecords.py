import tensorflow as tf

from pathlib import Path
feature_descriptions = {

    'image': tf.io.FixedLenFeature([], tf.string),

    'target_positions': tf.io.FixedLenFeature([], tf.string),

    'target_yaws': tf.io.FixedLenFeature([], tf.string),

    'target_availabilities': tf.io.FixedLenFeature([], tf.string),

    'history_positions': tf.io.FixedLenFeature([], tf.string),

    'history_yaws': tf.io.FixedLenFeature([], tf.string),

    'history_availabilities': tf.io.FixedLenFeature([], tf.string),

    'world_to_image': tf.io.FixedLenFeature([], tf.string),

    'track_id': tf.io.FixedLenFeature([], tf.string),

    'timestamp': tf.io.FixedLenFeature([], tf.string),

    'centroid': tf.io.FixedLenFeature([], tf.string),

    'yaw': tf.io.FixedLenFeature([], tf.string),

    'extent': tf.io.FixedLenFeature([], tf.string),

}



feature_dtypes = {

    'image': tf.float32,

    'target_positions': tf.float32,

    'target_yaws': tf.float32,

    'target_availabilities': tf.float32,

    'history_positions': tf.float32,

    'history_yaws': tf.float32,

    'history_availabilities': tf.float32,

    'world_to_image': tf.float64,

    'track_id': tf.int64,

    'timestamp': tf.int64,

    'centroid': tf.float64,

    'yaw': tf.float64,

    'extent': tf.float32,

}



def make_decoder(descriptions, dtypes):

    def decode_example(example):

        example_1 = tf.io.parse_single_example(example, descriptions)

        example_2 = []

        for key in dtypes.keys():

            example_2.append(

                tf.io.parse_tensor(example_1[key], dtypes[key])

            )

        return example_2

    return decode_example



decoder = make_decoder(feature_descriptions, feature_dtypes)
DATA_DIR = Path('../input/lyft-motion-prediction-tfrecords')



train_files = tf.io.gfile.glob(str(DATA_DIR / 'training' / 'training' / '*.tfrecord'))

valid_files = tf.io.gfile.glob(str(DATA_DIR / 'validation' / 'validation' / '*.tfrecord'))



ds_train = (

    tf.data.TFRecordDataset(train_files)

    .map(decoder)

    # etc

)



ds_valid = (

    tf.data.TFRecordDataset(valid_files)

    .map(decoder)

    # etc

)
import matplotlib.pyplot as plt
for batch in ds_train.take(1):

    image_0 = batch[0][0]

    pos_0 = batch[0][1]

    plt.figure(figsize=(16, 8))

    for channel in range(5):

        plt.subplot(2, 5, channel+1)

        plt.imshow(image_0[channel], cmap='gray')

        plt.subplot(2, 5, channel+1+5)

        plt.imshow(pos_0[channel], cmap='gray')
for batch in ds_valid.take(1):

    image_0 = batch[0][0]

    pos_0 = batch[0][1]

    plt.figure(figsize=(16, 8))

    for channel in range(5):

        plt.subplot(2, 5, channel+1)

        plt.imshow(image_0[channel], cmap='gray')

        plt.subplot(2, 5, channel+1+5)

        plt.imshow(pos_0[channel], cmap='gray')