import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



#IMAGE_DIR = "../input/celeba-dataset/img_align_celeba/img_align_celeba"
IMG_HEIGHT = 128 
IMG_WIDTH =  128
BATCH_SIZE = 32
AUTO = tf.data.experimental.AUTOTUNE


#The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):

  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
 return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_string):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  image_shape = tf.image.decode_jpeg(image_string).shape
  feature = {

      'image': _bytes_feature(image_string),
  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
def process_img(img):
    
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)

  # crop image  
  #credit: https://www.kaggle.com/tobirohrer/gan-with-tensorflow-and-tf-dataset
  img = tf.image.central_crop(img, 0.7)
  img = tf.image.crop_to_bounding_box(img, 
                                    offset_height = 30, 
                                    offset_width = 10, 
                                    target_height = 115, 
                                    target_width = 115
                                     )

  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.    
  return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])



#function to create tfrecord
def create_tfrecord(file_dir,name='celeba'):
    
    #tfrecord filename
    filename = name + '.tfrecord'

    with tf.io.TFRecordWriter(filename) as writer:

        for image in tqdm(os.listdir(file_dir)):
                
            img_string = open(os.path.join(file_dir,image), 'rb').read()
            #img = process_img(img)
            #img_string = base64.b64encode(img)
                            
            example = serialize_example(img_string)
            writer.write(example)

#process the image an save it to new directory in .jpg format
from PIL import Image

def process_and_save():
    
    for image_name in tqdm(os.listdir(IMAGE_DIR)):
        
        img_string = open(os.path.join(IMAGE_DIR,image_name), 'rb').read()
        img = process_img(img_string)
        img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
        img = img.save('celeba-new/'+image_name)


                
#save image
process_and_save()
#create tfrecord
create_tfrecord(file_dir='celeba-new')

# Create a dictionary describing the features.
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
  image = decode_image(parsed_example['image'])

  return image

def decode_image(image_data):
    #decode image 
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image,[IMG_HEIGHT,IMG_WIDTH])
    return image
#function to plot images
def view_image(ds):
    images = next(iter(ds)) # extract 1 batch from the dataset
    images = images.numpy()

    fig = plt.figure(figsize=(20, 20))
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])

#create tfdataset from tfrecord
TFRECORD_FILE = '../input/celeba-tfrecord/celeba.tfrecord'
def create_dataset():

    dataset = tf.data.TFRecordDataset(TFRECORD_FILE)

    dataset = dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.repeat().shuffle(1024).batch(BATCH_SIZE).prefetch(AUTO)
    
    return dataset
#create ds and view
ds = create_dataset()
view_image(ds)
