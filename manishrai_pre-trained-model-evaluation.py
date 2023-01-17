# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# !git clone https://github.com/tensorflow/models.git
# !pip install protobuf-compiler 

# !pip install python-pil 

# !pip install python-lxml 

# !pip install python-tk

# !pip install --user Cython

# !pip install --user contextlib2

# # !pip install --user jupyter

# !pip install --user matplotlib
# !export PYTHONPATH="${PYTHONPATH}:/kaggle/working/models/"

# !export PYTHONPATH="${PYTHONPATH}:/kaggle/working/models/research"

# !export PYTHONPATH="${PYTHONPATH}:/kaggle/working/models/research/slim/"
# %cd models/research/
# !wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip

# !unzip protobuf.zip
# From tensorflow/models/research/

# !./bin/protoc object_detection/protos/*.proto --python_out=.
import numpy as np

import os

import six.moves.urllib as urllib

import sys

import tarfile

import tensorflow as tf

import zipfile



from distutils.version import StrictVersion

from collections import defaultdict

from io import StringIO

from matplotlib import pyplot as plt

from PIL import Image



import glob



# This is needed since the notebook is stored in the object_detection folder.

sys.path.append("..")

from object_detection.utils import ops as utils_ops



if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):

    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
# This is needed to display the images.

%matplotlib inline
from object_detection.utils import label_map_util



from object_detection.utils import visualization_utils as vis_util
# What model to download.

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

MODEL_FILE = MODEL_NAME + '.tar.gz'

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'



# Path to frozen detection graph. This is the actual model that is used for the object detection.

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'



# List of the strings that is used to add correct label for each box.

PATH_TO_LABELS = os.path.join('./object_detection/data', 'mscoco_label_map.pbtxt')



opener = urllib.request.URLopener()

opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

tar_file = tarfile.open(MODEL_FILE)

for file in tar_file.getmembers():

    file_name = os.path.basename(file.name)

    if 'frozen_inference_graph.pb' in file_name:

        tar_file.extract(file, os.getcwd())
detection_graph = tf.Graph()

with detection_graph.as_default():

    od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:

        serialized_graph = fid.read()

        od_graph_def.ParseFromString(serialized_graph)

        tf.import_graph_def(od_graph_def, name='')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
def load_image_into_numpy_array(image):

    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape(

              (im_height, im_width, 3)).astype(np.uint8)



# PATH_TO_TEST_IMAGES_DIR = '../../../input/test/'

TEST_IMAGE_PATHS = glob.glob('../../../input/test/*')[20:21] #[ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]



# Size, in inches, of the output images.

IMAGE_SIZE = (12, 8)

TEST_IMAGE_PATHS
def run_inference_for_single_image(image, graph):

  with graph.as_default():

    with tf.Session() as sess:

      # Get handles to input and output tensors

      ops = tf.get_default_graph().get_operations()

      all_tensor_names = {output.name for op in ops for output in op.outputs}

      tensor_dict = {}

      for key in [

          'num_detections', 'detection_boxes', 'detection_scores',

          'detection_classes', 'detection_masks'

      ]:

        tensor_name = key + ':0'

        if tensor_name in all_tensor_names:

          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(

              tensor_name)

      if 'detection_masks' in tensor_dict:

        # The following processing is only for single image

        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])

        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.

        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)

        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])

        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(

            detection_masks, detection_boxes, image.shape[0], image.shape[1])

        detection_masks_reframed = tf.cast(

            tf.greater(detection_masks_reframed, 0.5), tf.uint8)

        # Follow the convention by adding back the batch dimension

        tensor_dict['detection_masks'] = tf.expand_dims(

            detection_masks_reframed, 0)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')



      # Run inference

      output_dict = sess.run(tensor_dict,

                             feed_dict={image_tensor: np.expand_dims(image, 0)})



      # all outputs are float32 numpy arrays, so convert types as appropriate

      output_dict['num_detections'] = int(output_dict['num_detections'][0])

      output_dict['detection_classes'] = output_dict[

          'detection_classes'][0].astype(np.uint8)

      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]

      output_dict['detection_scores'] = output_dict['detection_scores'][0]

      if 'detection_masks' in output_dict:

        output_dict['detection_masks'] = output_dict['detection_masks'][0]

  return output_dict
for image_path in TEST_IMAGE_PATHS:

    image = Image.open(image_path)

    # the array based representation of the image will be used later in order to prepare the

    # result image with boxes and labels on it.

    image_np = load_image_into_numpy_array(image)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.

    output_dict = run_inference_for_single_image(image_np, detection_graph)

    # Visualization of the results of a detection.

    vis_util.visualize_boxes_and_labels_on_image_array(

          image_np,

          output_dict['detection_boxes'],

          output_dict['detection_classes'],

          output_dict['detection_scores'],

          category_index,

          instance_masks=output_dict.get('detection_masks'),

          use_normalized_coordinates=True,

          line_thickness=2, 

          min_score_thresh=0.2)

    plt.figure(figsize=IMAGE_SIZE)

    plt.imshow(image_np)
def format_prediction_string(image_id, result):

    prediction_strings = []

    

    for i in range(2):#range(len(result['detection_scores'])):

#         category_index[result['detection_classes'][i]]['name']

        class_name = category_index[result['detection_classes'][i]]['name']#.decode("utf-8")

        boxes = result['detection_boxes'][i]

        score = result['detection_scores'][i]

        

        prediction_strings.append(

            f"{class_name} {score} " + " ".join(map(str, boxes))

        )

        

    prediction_string = " ".join(prediction_strings)



    return {

        "ImageID": image_id,

        "PredictionString": prediction_string

    }



# dict_keys(['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'])
from tqdm import tqdm



sample_submission_df = pd.read_csv('../../../input/sample_submission.csv')

image_ids = sample_submission_df['ImageId'][1:2]

predictions = []



for image_id in tqdm(image_ids):

    # Load the image string

    image_path = f'../../../input/test/{image_id}.jpg'

    image = Image.open(image_path)

    # result image with boxes and labels on it.

    image_np = load_image_into_numpy_array(image)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.

    output_dict = run_inference_for_single_image(image_np, detection_graph)

    predictions.append(format_prediction_string(image_id, output_dict))
pred_df = pd.DataFrame(predictions)

pred_df.head()
pred_df.to_csv('../../../submission.csv', index=False)
# !kg competitions submit -c 'open-images-2019-object-detection' -f submission.csv -m "Test Submission"