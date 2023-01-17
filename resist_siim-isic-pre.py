import tensorflow as tf
DIR_PATH = "../input/siim-isic-melanoma-classification"

JPEG_PATH = f"{DIR_PATH}/jpeg/train"

IMAGE_SIZE = 256

N_FILES = 25

DATA_CARD = 33126
def _serialize_example(image, target):

    """Writes the image and target to a protobuf."""

    feature = {

        "image": tf.train.Feature(

            bytes_list=tf.train.BytesList(value=[image.numpy()])),

        "target": tf.train.Feature(int64_list=tf.train.Int64List(value=[target]))

    }

    

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()

                                   

def load_resize_and_serialize(image_name, target):

    image_file = tf.strings.join([image_name, "jpg"], separator=".")

    

    read_image = tf.io.read_file(

        tf.strings.join([JPEG_PATH, image_file], separator="/"))

    image = tf.io.decode_jpeg(read_image, channels=3)

                                                # h         # w

    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE,

                                     method="lanczos5", antialias=True)

    serialized_image = tf.io.serialize_tensor(tf.cast(image, tf.uint8))

    

    return tf.py_function(_serialize_example, (serialized_image, target), tf.string)
data_pipe = tf.data.experimental.CsvDataset(

    f"{DIR_PATH}/train.csv",

    [tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.int64)],

    header=True, select_cols=[0, 7]

).shuffle(

    10000, seed=1236, reshuffle_each_iteration=False

).map(

    load_resize_and_serialize,

    num_parallel_calls=tf.data.experimental.AUTOTUNE,

    deterministic=False

).window(DATA_CARD // N_FILES, drop_remainder=True

).enumerate(

).prefetch(tf.data.experimental.AUTOTUNE)



for n, data in data_pipe:

    if n < 20:

        file_name = tf.strings.format("train_{}_{}.tfrecord", (n % 5, n % 4))

    else:

        file_name = tf.strings.format("valid_{}.tfrecord", n % 5)

    

    tf.data.experimental.TFRecordWriter(

        file_name, compression_type="GZIP"

    ).write(data)