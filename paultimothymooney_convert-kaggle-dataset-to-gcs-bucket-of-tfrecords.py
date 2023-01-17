import os

import math

import numpy as np

from matplotlib import pyplot as plt

import tensorflow as tf

from google.cloud import storage
# For uploading to GCS buckets:

STORAGE_CLIENT = storage.Client(project='kaggle-playground-170215')

# For converting images to TFRecord files:

WORKING_DIRECTORY = "./"

BASE_DIR = '/kaggle/input/flowers-recognition/flowers/flowers'

FILE_PATTERN = BASE_DIR + '/*/*.jpg'

TARGET_SIZE = [512, 512]

CLASSES = os.listdir(BASE_DIR)

AUTO = tf.data.experimental.AUTOTUNE

SHARDS = 16



def create_bucket(dataset_name):

    """Creates a new bucket. https://cloud.google.com/storage/docs/ """

    bucket = STORAGE_CLIENT.create_bucket(dataset_name)

    print('Bucket {} created'.format(bucket.name))



def upload_blob(bucket_name, source_file_name, destination_blob_name):

    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """

    bucket = STORAGE_CLIENT.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(

        source_file_name,

        destination_blob_name))

    

def list_blobs(bucket_name):

    """Lists all the blobs in the bucket. https://cloud.google.com/storage/docs/"""

    blobs = STORAGE_CLIENT.list_blobs(bucket_name)

    for blob in blobs:

        print(blob.name)

        

def download_to_kaggle(bucket_name,destination_directory,file_name):

    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""

    os.makedirs(destination_directory, exist_ok = True)

    full_file_path = os.path.join(destination_directory, file_name)

    blobs = STORAGE_CLIENT.list_blobs(bucket_name)

    for blob in blobs:

        blob.download_to_filename(full_file_path)

    

def decode_jpeg_and_label(filename):

  bits = tf.io.read_file(filename)

  image = tf.image.decode_jpeg(bits)

  label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')

  label = label.values[-2]

  return image, label



def resize_and_crop_image(image, label):

  # Resize and crop using "fill" algorithm:

  # always make sure the the resulting image

  # is cut out from the source image so that

  # it fills the TARGET_SIZE entirely with no

  # black bars and a preserved aspect ratio.

  w = tf.shape(image)[0]

  h = tf.shape(image)[1]

  tw = TARGET_SIZE[1]

  th = TARGET_SIZE[0]

  resize_crit = (w * th) / (h * tw)

  image = tf.cond(resize_crit < 1,

                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true

                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false

                 )

  nw = tf.shape(image)[0]

  nh = tf.shape(image)[1]

  image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)

  return image, label



def recompress_image(image, label):

  height = tf.shape(image)[0]

  width = tf.shape(image)[1]

  image = tf.cast(image, tf.uint8)

  image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

  return image, label, height, width



# Three types of data can be stored in TFRecords: bytestrings, integers and floats

# They are always stored as lists, a single data element will be a list of size 1



def _bytestring_feature(list_of_bytestrings):

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))



def _int_feature(list_of_ints): # int64

  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))



def _float_feature(list_of_floats): # float32

  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

  

def to_tfrecord(tfrec_filewriter, img_bytes, label, height, width):  

  class_num = np.argmax(np.array(CLASSES)==label) # 'roses' => 2 (order defined in CLASSES)

  one_hot_class = np.eye(len(CLASSES))[class_num]     # [0, 0, 1, 0, 0] for class #2, roses

  feature = {

      "image": _bytestring_feature([img_bytes]), # one image in the list

      "class": _int_feature([class_num]),        # one class in the list

      # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data

      "label":         _bytestring_feature([label]),          # fixed length (1) list of strings, the text label

      "size":          _int_feature([height, width]),         # fixed length (2) list of ints

      "one_hot_class": _float_feature(one_hot_class.tolist()) # variable length  list of floats, n=len(CLASSES)

  }

  return tf.train.Example(features=tf.train.Features(feature=feature))



def read_tfrecord(example):

    features = {

        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)

        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar

        

        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data

        "label":         tf.io.FixedLenFeature([], tf.string),  # one bytestring

        "size":          tf.io.FixedLenFeature([2], tf.int64),  # two integers

        "one_hot_class": tf.io.VarLenFeature(tf.float32)        # a certain number of floats

    }

    # decode the TFRecord

    example = tf.io.parse_single_example(example, features)

    # FixedLenFeature fields are now ready to use: exmple['size']

    # VarLenFeature fields require additional sparse_to_dense decoding

    image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.reshape(image, [*TARGET_SIZE, 3])

    class_num = example['class']

    label  = example['label']

    height = example['size'][0]

    width  = example['size'][1]

    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])

    return image, class_num, label, height, width, one_hot_class



def display_9_images_from_dataset(dataset):

  plt.figure(figsize=(13,13))

  subplot=331

  for i, (image, label) in enumerate(dataset):

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(image.numpy().astype(np.uint8))

    plt.title(label.numpy().decode("utf-8"), fontsize=16)

    subplot += 1

    if i==8:

      break

  plt.tight_layout()

  plt.subplots_adjust(wspace=0.1, hspace=0.1)

  plt.show()
def convert_image_dataset_to_tfrecords_format(BASE_DIR,FILE_PATTERN,

                                              TARGET_SIZE,CLASSES,

                                              WORKING_DIRECTORY,

                                              SHARDS,AUTO):

    '''

    Converts an image dataset into TFRecords format.

    Note that the image dataset should be organized

    such that different classes of images in different 

    folders.

    '''

    nb_images = len(tf.io.gfile.glob(FILE_PATTERN))

    shard_size = math.ceil(1.0 * nb_images / SHARDS)

    #print("Pattern matches {} images which will be rewritten as {} .tfrec files containing {} images each.".format(nb_images, SHARDS, shard_size))

    filenames = tf.data.Dataset.list_files(FILE_PATTERN, seed=35155) # This also shuffles the images

    dataset1 = filenames.map(decode_jpeg_and_label, num_parallel_calls=AUTO)

    #display_9_images_from_dataset(dataset1)

    dataset2 = dataset1.map(resize_and_crop_image, num_parallel_calls=AUTO)  

    #display_9_images_from_dataset(dataset2)

    dataset3 = dataset2.map(recompress_image, num_parallel_calls=AUTO)

    dataset3 = dataset3.batch(shard_size) # sharding: there will be one "batch" of images per file 

    #print("Writing TFRecords")

    for shard, (image, label, height, width) in enumerate(dataset3):

      # batch size used as shard size here

      shard_size = image.numpy().shape[0]

      # good practice to have the number of records in the filename

      filename = WORKING_DIRECTORY + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:

        for i in range(shard_size):

          example = to_tfrecord(out_file,

                                image.numpy()[i], # re-compressed image: already a byte string

                                label.numpy()[i],

                                height.numpy()[i],

                                width.numpy()[i])

          out_file.write(example.SerializeToString())

        #print("Wrote file {} containing {} records".format(filename, shard_size))

    # read from TFRecords. For optimal performance, use "interleave(tf.data.TFRecordDataset, ...)"

    # to read from multiple TFRecord files at once and set the option experimental_deterministic = False

    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()

    option_no_order.experimental_deterministic = False

    dataset4 = tf.data.Dataset.list_files(WORKING_DIRECTORY + "*.tfrec")

    dataset4 = dataset4.with_options(option_no_order)

    dataset4 = dataset4.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)

    dataset4 = dataset4.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset4 = dataset4.shuffle(300)

    display_dataset = dataset4.map(lambda image, class_num, label, height, width, one_hot_class: (image, label))

    #display_9_images_from_dataset(display_dataset)

    return dataset4
convert_image_dataset_to_tfrecords_format(BASE_DIR,FILE_PATTERN,

                                              TARGET_SIZE,CLASSES,

                                              WORKING_DIRECTORY,

                                              SHARDS,AUTO)

#!rm '__notebook_source__.ipynb'

#!rm -r '.ipynb_checkpoints'



list_of_files = os.listdir('/kaggle/working/')

print(list_of_files)
bucket_name = 'flowers_dataset_1'         

try:

    create_bucket(bucket_name)   

except:

    pass
for file in list_of_files:

    local_data = WORKING_DIRECTORY+file

    file_name = file

    upload_blob(bucket_name, local_data, file_name)



print('\nData inside of the GCS Bucket ',bucket_name,':\n')

list_blobs(bucket_name)  
# os.listdir('/kaggle/working/flowers/')

    

# # FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/flowers/'
destination_directory = '/kaggle/working/flowers/'       

for file_name in list_of_files:

    download_to_kaggle(bucket_name,destination_directory,file_name)
os.listdir('/kaggle/working/flowers/')