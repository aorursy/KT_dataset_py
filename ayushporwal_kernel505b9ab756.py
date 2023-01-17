import tensorflow as tf

from kaggle_datasets import KaggleDatasets

AUTO = tf.data.experimental.AUTOTUNE
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print(f'Running on: {tpu.master()}')

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print(f'REPLICAS: {strategy.num_replicas_in_sync}')
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
IMAGE_SIZE = [1024, 1024]

EPOCHS = 100

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/train*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*.tfrec')

def decode_image(example):

    image = tf.image.decode_jpeg(example, channels = 3)

    image = tf.cast(image, tf.float32) / 255

    image = tf.reshape(image, [1, *IMAGE_SIZE, 3])

    return image





def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'image_name': tf.io.FixedLenFeature([], tf.string),

        'target': tf.io.FixedLenFeature([], tf.int64)

    }

    

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.float32)

    image_name = tf.cast(example['image_name'], tf.string)

    return image, label, image_name





def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'image_name': tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    image_name = tf.cast(example['image_name'], tf.string)

    return image, image_name



def load_dataset(filenames, labeled=True, ordered = False):

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False

    

    

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset.with_options(ignore_order)

    dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    return dataset



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2071)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_test_dataset():

    dataset = load_dataset(TEST_FILENAMES, labeled=False)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2071)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



DATASET_SIZE = 33126

STEPS_PER_EPOCH = DATASET_SIZE//BATCH_SIZE
with strategy.scope():

    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(50,3,input_shape = [*IMAGE_SIZE, 3]),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(25, activation = 'tanh'),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(12, activation = 'sigmoid'),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(2, activation='softmax')

    ])



model.compile(

    optimizer='adam',

    loss = 'mse',

    metrics=['mae']

)



model.summary()
history = model.fit(get_training_dataset(), steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS)