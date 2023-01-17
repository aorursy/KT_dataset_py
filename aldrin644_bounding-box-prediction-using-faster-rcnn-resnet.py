import os
os.chdir('../input')
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from mods_folder import ops as utils_ops

import json

with open('mods_folder/old_oid_labels.json') as f:
    old_oid = json.load(f)
 
# This is needed to display the images.
%matplotlib inline
#MODEL_NAME = 'r-faster-cnn-inception-v2'
MODEL_NAME = 'slow-rcnn'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = 'mods_folder/oid_bbox_trainable_label_map.pbtxt'

NUM_CLASSES = 545
from mods_folder import label_map_util

from mods_folder import visualization_utils as vis_util
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
os.listdir('google-ai-open-images-object-detection-track/test/challenge2018_test')
PATH_TO_TEST_IMAGES_DIR = 'google-ai-open-images-object-detection-track/test/challenge2018_test'

TEST_IMAGE_PATHS = os.listdir('google-ai-open-images-object-detection-track/test/challenge2018_test')   

TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[:9]

IMAGE_SIZE = (12, 8)
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
prediction_list = []
for image_name in TEST_IMAGE_PATHS:
  image_path = os.path.join('google-ai-open-images-object-detection-track/test/challenge2018_test',image_name)  
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
    
  # Visualization of the results of a detection.
  detection_boxes = np.around(output_dict['detection_boxes'],decimals=2)
  detection_classes = output_dict['detection_classes']
  detection_scores = np.around(output_dict['detection_scores'],decimals=1)  
  ind = np.where(output_dict['detection_boxes'].any(axis=1))[0]
  ind = list(ind)

  def get_class(id_num):
        class_label = ''
        for i in old_oid:
            if(i['id'] == id_num):
                class_label = i['name']
                break
        return class_label        
  def get_class_name(id_num):
        class_name = ''
        for i in old_oid:
            if(i['id'] == id_num):
                class_name = i['display_name']
                break
        return class_name 
        
                
  pred_str = ''      
  pred_name = ''
  for i in ind: 
      l = output_dict['detection_boxes'][i]
      id_num = output_dict['detection_classes'][i]
      cls_label = get_class(id_num)
      cls_name = get_class_name(id_num)  
      prob = str('{:.4f}'.format(output_dict["detection_scores"][i]))
      pred_name = pred_name + cls_name + ' ' + prob + ' | '  
      bounding_box = ' '.join([str('{:.4f}'.format(w))+' ' for w in l])  
      pred_str = pred_str + cls_label + ' ' + prob + ' ' + bounding_box + ' '
    
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure(figsize=IMAGE_SIZE)  
  plt.imshow(image_np)
  plt.show()
  prediction_object = {'ImageId':image_name[:-4], 'PredictionString':pred_str}  
  print('Detected Classes =>')
  print(pred_name[:-1])
  print()
  print(prediction_object)
  prediction_list.append(prediction_object)  

df = pd.DataFrame.from_dict(prediction_list, orient='columns')
df.to_csv('../working/prediction.csv', index=False)