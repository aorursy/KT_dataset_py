# tf.keras.preprocessing.image_dataset_from_directory will be available in TF2.3

!pip install -q tf-nightly



import tensorflow as tf
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file(origin=dataset_url, 

                                   fname='flower_photos', 

                                   untar=True,

                                   cache_dir="/kaggle/working/downloaded")

data_dir = pathlib.Path(data_dir)
from functools import partial



IMG_HEIGHT = 512

IMG_WIDTH = 512



load_split = partial(

    tf.keras.preprocessing.image_dataset_from_directory,

    data_dir,

    validation_split=0.2,

    shuffle=True,

    seed=123,

    image_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=1,

)



ds_train = load_split(subset='training')

ds_valid = load_split(subset='validation')



class_names = ds_train.class_names

print("\nClass names: {}".format(class_names))
from tensorflow.train import BytesList, FloatList, Int64List

from tensorflow.train import Example, Features, Feature



def process_image(image, label):

    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

    image = tf.io.encode_jpeg(image)

    return image, label



ds_train_encoded = (

    ds_train

    .unbatch()

    .map(process_image)

)



ds_valid_encoded = (

    ds_valid

    .unbatch()

    .map(process_image)

)
def make_example(encoded_image, label):

    image_feature = Feature(

        bytes_list=BytesList(value=[

            encoded_image,

        ]),

    )

    label_feature = Feature(

        int64_list=Int64List(value=[

            label,

        ])

    )



    features = Features(feature={

        'image': image_feature,

        'label': label_feature,

    })

    

    example = Example(features=features)

    

    return example.SerializeToString()
!mkdir '/kaggle/working/training'



NUM_SHARDS = 32

PATH = '/kaggle/working/training/shard_{:02d}.tfrecord'



for shard in range(NUM_SHARDS):

    ds_shard = (

        ds_train_encoded

        .shard(NUM_SHARDS, shard)

        .as_numpy_iterator()

    )

    with tf.io.TFRecordWriter(path=PATH.format(shard)) as f:

        for encoded_image, label in ds_shard:

            example = make_example(encoded_image, label)

            f.write(example)
!mkdir '/kaggle/working/validation'



NUM_SHARDS = 8

PATH = '/kaggle/working/validation/shard_{:02d}.tfrecord'



for shard in range(NUM_SHARDS):

    ds_shard = (

        ds_valid_encoded

        .shard(NUM_SHARDS, shard)

        .as_numpy_iterator()

    )

    with tf.io.TFRecordWriter(path=PATH.format(shard)) as f:

        for encoded_image, label in ds_shard:

            example = make_example(encoded_image, label)

            f.write(example)
!rm -rf '/kaggle/working/downloaded'