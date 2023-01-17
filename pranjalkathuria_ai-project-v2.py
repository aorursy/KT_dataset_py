# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import os

import six.moves.urllib as urllib

import sys

import tarfile

import tensorflow as tf

import zipfile

import pathlib



from collections import defaultdict

from io import StringIO

from matplotlib import pyplot as plt

from PIL import Image

from IPython.display import display

!pip install tf_slim
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
# Environment Variables

os.environ['AUTOGRAPH_VERBOSITY'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['PYTHONPATH']=os.environ['PYTHONPATH']+':/kaggle/models/research/slim:/kaggle/models/research'

os.environ['PYTHONPATH']
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
# patch tf1 into `utils.ops`

utils_ops.tf = tf.compat.v1



# Patch the location of gfile

tf.gfile = tf.io.gfile
def load_model(model_name):

  base_url = 'http://download.tensorflow.org/models/object_detection/'

  model_file = model_name + '.tar.gz'

  model_dir = tf.keras.utils.get_file(

    fname=model_name, 

    origin=base_url + model_file,

    untar=True)



  model_dir = pathlib.Path(model_dir)/"saved_model"



  model = tf.saved_model.load(str(model_dir))

  model = model.signatures['serving_default']



  return model
# List of the strings that is used to add correct label for each box.

#PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'

PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/kaggle/input')

TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

TEST_IMAGE_PATHS
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

detection_model = load_model(model_name)
print(detection_model.inputs)
detection_model.output_dtypes
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

  output_dict = {key:value[0, :num_detections].numpy() 

                 for key,value in output_dict.items()}

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
for image_path in TEST_IMAGE_PATHS:

  show_inference(detection_model, image_path)