!/opt/conda/bin/python3.7 -m pip install --upgrade pip

!pip install tf_slim

!pip install pycocotools

!pip install --user Cython -q

!pip install --user contextlib2 -q

!pip install --user pillow -q

!pip install --user lxml -q
import pdb

import pathlib

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import pandas as pd

import tensorflow as tf

import sys

import tarfile

import tempfile

import zipfile

import glob

import cv2

from pathlib import Path



from PIL import Image, ImageOps

from IPython.display import display

import hvplot.pandas

import os



%matplotlib inline

%xmode Verbose



image_path = Path('/kaggle/input/open-images-object-detection-rvc-2020')

data_path = Path('/kaggle/input/open-image-2019')

image_list = sorted(image_path.glob('test/*.jpg'))



print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    # Restrict TensorFlow to only use the first GPU

    try:

        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

    except RuntimeError as e:

        # Visible devices must be set before GPUs have been initialized

        print(e)
print('Getting image ids...')

df_image_ids = pd.read_csv(image_path/'sample_submission.csv')

df_image_ids.drop('PredictionString', axis=1, inplace=True)
# Install the protobufs compiler

!wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip -q

!unzip -o protobuf.zip

!rm protobuf.zip
%cd /kaggle

!rm -fr models

!git clone https://github.com/tensorflow/models.git

!rm -fr models/.git
# Compile protobufs

%cd /kaggle/models/research

!../../working/bin/protoc object_detection/protos/*.proto --python_out=.
# Install models

!pip install .
# Environment Variables

os.environ['AUTOGRAPH_VERBOSITY'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['PYTHONPATH']=os.environ['PYTHONPATH']+':/kaggle/models/research/slim:/kaggle/models/research'

os.environ['PYTHONPATH']
# Import object detection model

from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
# patch tf1 into `utils.ops`

utils_ops.tf = tf.compat.v1



# Patch the location of gfile

tf.gfile = tf.io.gfile
!python object_detection/builders/model_builder_test.py
# Get the file name from the image id

def filename_from_id(id):

    return os.path.join(image_path, 'test/', '{}.jpg'.format(id) )
# Resizes image to new_width x new_height and returns PIL file

def resize_image(path, new_width=900, new_height=900):

    _, filename = tempfile.mkstemp(suffix=".jpg")

    pil_image = Image.open(path)

    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)

    pil_image_rgb = pil_image.convert("RGB")

    pil_image_rgb.save(filename, format="JPEG", quality=90)

    # print('Resized image saved as: {}'.format(filename))

    return filename
# Display a PIL image file

def display_image(image_path):

    image_np = np.array(Image.open(image_path))

    display(Image.fromarray(image_np))
%cd /kaggle

sample_submission_df = pd.read_csv(f'{image_path}/sample_submission.csv')

image_ids = sample_submission_df['ImageId']

del sample_submission_df
test_img = 50000

# Build a list of images



filename = filename_from_id(image_ids[test_img])



# Load, resize and display sample image

filename_r = resize_image(filename)

print(filename_r)

display_image(filename_r)

# Model Loader

def load_model(model_name):

    base_url = 'http://download.tensorflow.org/models/object_detection/'

    model_file = model_name + '.tar.gz'

    model_dir = tf.keras.utils.get_file(fname=model_name, 

                                        origin=base_url + model_file,

                                        untar=True)



    model_dir = pathlib.Path(model_dir)/"saved_model"



    model = tf.saved_model.load(str(model_dir))

    model = model.signatures['serving_default']



    return model
# Load Label Map

PATH_TO_LABELS = 'models/research/object_detection/data/oid_v4_label_map.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'

detection_model = load_model(model_name)
def run_inference_for_single_image(model, image):

    image = np.asarray(image)

    

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.

    input_tensor = tf.convert_to_tensor(image)

    

    # The model expects a batch of images, so add an axis with `tf.newaxis`.

    input_tensor = input_tensor[tf.newaxis,...]



    # Run inference

    output_dict = model(input_tensor)



    # All outputs are batches tensors.

    # Convert to numpy arrays, and take index [0] to remove the batch dimension.

    # We're only interested in the first num_detections.

    num_detections = int(output_dict.pop('num_detections'))

    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}

    output_dict['num_detections'] = num_detections



    # detection_classes should be ints.

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

   

    # Handle models with masks:

    if 'detection_masks' in output_dict:

        # Reframe the the bbox mask to the image size.

        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(

                  output_dict['detection_masks'], output_dict['detection_boxes'],

                   image.shape[0], image.shape[1])      

        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,

                                           tf.uint8)

        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()



    return output_dict
def show_inference(model, image_path):

    # the array based representation of the image will be used later in order to prepare the

    # result image with boxes and labels on it.

    image_np = np.array(Image.open(image_path))

  

    # Actual detection.

    output_dict = run_inference_for_single_image(model, image_np)

    

    # Visualization of the results of a detection.

    vis_util.visualize_boxes_and_labels_on_image_array(

        image_np,

        output_dict['detection_boxes'],

        output_dict['detection_classes'],

        output_dict['detection_scores'],

        category_index,

        instance_masks=output_dict.get('detection_masks_reframed', None),

        use_normalized_coordinates=True,

        line_thickness=8)

    

    display(Image.fromarray(image_np))

    return output_dict
# Check the input signature and output types

print('Input signature: {}\n'.format(detection_model.inputs))

print('Output dtypes: {}\n'.format(detection_model.output_dtypes))

print('Output shapes: {}'.format(detection_model.output_shapes))
pred = show_inference(detection_model, filename_r)
print(pred)