!pip install tensorflow==1.15.0
import tensorflow as tf
print(tf.__version__)
import glob
import os
import cv2
import sys
import numpy as np
import pandas as pd
%matplotlib inline
from matplotlib import pyplot as plt
from pathlib import Path
import json
from shutil import copyfile
from distutils.dir_util import copy_tree
!git clone https://github.com/zeynepCankara/Clothing-Style-Detector.git
!git clone https://github.com/tensorflow/models.git
!git clone https://www.github.com/matterport/Mask_RCNN.git
os.chdir('Mask_RCNN')

!rm -rf .git # to prevent an error when the kernel is committed
!rm -rf images assets # to prevent displaying images at the bottom of a kernel

# copy subdirectory example
fromDirectory = "/kaggle/working/Clothing-Style-Detector"
toDirectory = "/kaggle/working/models/research/object_detection"

copy_tree(fromDirectory, toDirectory)

%cd /kaggle/working/models/research
!protoc object_detection/protos/*.proto --python_out=.

##########################################################################################################
################################  USE OPS.PY DOWNLOADED ON THE SYSTEM ####################################
##########################################################################################################
copyfile("/kaggle/input/additional/ops.py","/kaggle/working/models/research/object_detection/utils/ops.py")
sys.path.append('/kaggle/working/Mask_RCNN/')
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

glob_list = glob.glob(f'/kaggle/input/training-mask-r-cnn-to-be-a-fashionista-lb-0-07/fashion20190522T1516/mask_rcnn_fashion_0008.h5')
model_path = glob_list[0] if glob_list else ''
NUM_CATS = 46
IMAGE_SIZE = 512
ROOT_DIR = Path('/kaggle/working')

class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200
    
config = FashionConfig()
config.display()
###########################################################################################################################
class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model_seg = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model_seg.load_weights(model_path, by_name=True)
############################################################################################################################
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)  
    return img

##################################################################################################
########################     UPLOAD label_description.json      ##################################
##################################################################################################


with open("/kaggle/input/labels/label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]
def infer(dir,size):
    results = []
    for image in os.listdir(dire):
#         img = resize_image(str(dire+image))
#         result = model.detect([resize_image(str(dire+image))])
#################################################################################
        img = cv2.imread(str(dir+image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA) 
        result = model_seg.detect([img])
#################################################################################
        r = result[0]

        if r['masks'].size > 0:
            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            y_scale = img.shape[0]/256
            x_scale = img.shape[1]/256
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

            masks, rois = refine_masks(masks, rois)
        else:
            masks, rois = r['masks'], r['rois']
#         print(image)
        visualize.display_instances(img, rois, masks, r['class_ids'], 
                                    ['bg']+label_names, r['scores'],
                                    title=image, figsize=(12, 12))
        result[0]['name'] =  image
        results.append(result)
    return results
"""
Created on Sun Jul  1 08:56:06 2018
@author: Zeynep CANKARA
Detection module
"""
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
#make a prediction from your test
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.python.framework import ops
ops.reset_default_graph()

"""
    Read the json data for test
json_data=open('data.txt').read()
data = json.loads(json_data)
print(data)
"""

"""
    Function for cropping the image and detecting the clothing type, pattern, color on the image if exist
    the images will later saved with the bounding box and prediction 
    param: path, type = str, filepath of the image,
        xmin, ymin, xmax, ymax, type = int, bounding-box coordinates,
        im_width, im_height, type = int, image dimensions width x height
"""

def crop_image(path,xmin, ymin, xmax, ymax, im_width, im_height):
    myImage = cv.imread(str(path))
    myImage = cv.resize(myImage,(im_width,im_height))
    cropped = myImage[ymin: ymax, xmin:xmax]
    cv.imwrite("prediction/img_trial2.jpg", cropped)
    croped_predictions = clothing_color_pattern("prediction/img_trial2.jpg")
    #display the predictions on the console
    print(croped_predictions)
    if(str(croped_predictions) != ""):
        custom_bbox(str(path),str(croped_predictions) ,xmin ,  ymin, xmax, ymax, im_width, im_height)
    return croped_predictions

"""
    Function for drawing a bounding box on the image
    param: path, type = str, filepath of the image,
        predictions, type = str, 
        xmin, ymin, xmax, ymax, type = int, bounding-box coordinates,
        im_width, im_height, type = int, image dimensions width x height        
"""
def custom_bbox(path, predictions, xmin, ymin, xmax, ymax, im_width, im_height):
    img = cv.imread(str(path))
    img = cv.resize(img, (im_width, im_height))
    print(im_width)
    print(im_height)
    #drawing a rectangle on the image on the place where clothes detected
    img = cv.rectangle(img,(xmin,ymin),(xmax,ymax), (0,255,0), 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    #font size set according to the image size
    font_size = [1, 0.75, 0.5, 0.25, 0.10, 0.05]
    if((im_width * im_height) > 1638400 ):
        font_size = float(font_size[0])
    elif((im_width * im_height) > 409600):
        font_size = float(font_size[1])    
    elif((im_width * im_height) > 102400):
        font_size = float(font_size[2])
    elif((im_width * im_height) > 25600):
        font_size = float(font_size[3])
    elif((im_width * im_height) > 6400):
        font_size = float(font_size[4])  
    else:
        font_size = float(font_size[5])
    #writing the detection result on the bounding-box
    cv.putText(img,str(predictions),(int(xmin),int(ymax)), font, float(font_size) ,(0,0,0),2,cv.LINE_AA)
    cv.imwrite(str(path), img)

"""
    Function which loads models and performs the detection on the cropped section of the image
    param: path, type = str
"""

def clothing_color_pattern(path):
    #the dictionary for evaluation
    valid_classes = {'T-shirt': ['jersey', 'T-shirt', 'tee shirt'], 'Dress':['dress', 'gown', 'overskirt', 'hoopskirt', 'stole', 'abaya', 'academic_gown', 'poncho', 'breastplate'], 'Outerwear':['jacket', 'raincoat', 'trench coat','book jacket', 'dust cover', 'dust jacket', 'dust wrapper', 'pitcher'], 'Suit':['suit','bow tie', 'bow-tie', 'bowtie','suit of clothes'], 'Shirt':['shirt'], 'Sweater':['sweater', 'sweatshirt','bulletproof_vest', 'velvet'] , 'Tank top':['blause', 'tank top', 'maillot', 'bikini', 'two-piece', 'swimming trunks', 'bathing trunks'], 'Skirt':['miniskirt', 'mini']}
    have_glasses = {'Glasses': ['glasses', 'sunglass', 'sunglasses', 'dark glasses','shades']}
    wear_necklace = {'Necklace': ['neck_brace','necklace']}
    
    #initializing the prediictions
    prediction_color_clothes = ""
    acsessories = ""
    clothing_type = ""
    
    #LOADING MODALS
    #Load the ResNet50 model
    resnet_model = resnet50.ResNet50(weights='imagenet')
    #load pattern model
    pattern_model = load_model('/kaggle/working/Clothing-Style-Detector/weights/pattern.h5')
    #load color model
    color_model = load_model('/kaggle/working/Clothing-Style-Detector/weights/color.h5')
    
    #run model for color resNet class detection:
     #process for the resNet model
    test_image_resnet = image.load_img(path, target_size = (224, 224))
    test_image_resnet = image.img_to_array(test_image_resnet)
    
    #plot the image for test
    plt.imshow(test_image_resnet/255.)
    
    test_image_resnet = np.expand_dims(test_image_resnet, axis = 0)
    result_resnet = resnet_model.predict(test_image_resnet)
    label = decode_predictions(result_resnet)

    #predictions by resnet
    print(label[0])
    print(label[0][0])
    #check is prediction matches
    for element in range(len(label[0])):
        for key in valid_classes:
            if(label[0][element][1] in valid_classes[key]):
                if(float(label[0][element][2]) >= 0.055):
                    if(clothing_type == ""):
                        clothing_type = str(key)
                        break
                    
    #check for acsessories
    for element in range(len(label[0])):
        for key in have_glasses:
            if(label[0][element][1] in have_glasses[key]):
                if(float(label[0][element][2]) >= 0.04):
                    acsessories += str(key) + ","
    
    for element in range(len(label[0])):
        for key in wear_necklace:
            if(label[0][element][1] in wear_necklace[key]):
                if(float(label[0][element][2]) >= 0.05):
                    acsessories += str(key) + " "   
    
    #prepare the input image
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #predicting the pattern and color
    result_pattern = pattern_model.predict_classes(test_image)
    result_color = color_model.predict_classes(test_image)

                    
    #check the pattern   
    pattern_classes = ['Floral','Graphics','Plaid','Solid','Spotted','Striped']            
    prediction_pattern = pattern_classes[int(result_pattern)]
    
    #check the color
    color_classes =  ['Black', 'Blue', 'Brown', 'Cyan', 'Gray', 'Green', 'More than 1 color', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    prediction_color = color_classes[int(result_color)]


    #add the pattern info to the prediction
    prediction_color_clothes += str(prediction_pattern) + " , " + str(prediction_color)
    
    if((acsessories == "") and (clothing_type == "")):
        return str(prediction_color_clothes)
    else:
        return str(acsessories) + " " + str(prediction_color_clothes) + " " + str(clothing_type)



"""
    Takes the prediction of the acsessories if the class prediction in the acsessories 
    param: acsessories_class prediction of the acsessoies in the acsessories dictionary
    outputs the prediction, later to be used in custom_bbox()
    
    things to note: acsessories classes do not match with imagenet classes but match with 
    pbtxt file which you direct your model
    you can change this if you change type of your model from (model_zoo)
    COCO is good at people detection
"""

def acsessory_pattern_color(acsessories_class, path, xmin, ymin, xmax, ymax, im_width, im_height):
    myImage = cv.imread(str(path))
    myImage = cv.resize(myImage,(im_width,im_height))
    cropped = myImage[ymin: ymax, xmin:xmax]
    cv.imwrite("prediction/img_trial2.jpg", cropped)
    #load pattern model
    pattern_model = load_model('pattern2.h5')
    #load color model
    color_model = load_model('color.h5')
    #Process for custom_color_classifier
    test_image = image.load_img(str("prediction/img_trial2.jpg"), target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    #predicting the pattern
    result_pattern = pattern_model.predict_classes(test_image)
    #result_color = color_model.predict_classes(test_image)
    
    #process for custom_pattern_classifier
    test_image2 = image.load_img(str("prediction/img_trial2.jpg"), target_size = (128, 128))
    test_image2 = image.img_to_array(test_image2)
    test_image2 = np.expand_dims(test_image2, axis = 0)    
    result_color = color_model.predict_classes(test_image2)
    #check the color
    color_classes =  ['Black', 'Blue', 'Brown', 'Cyan', 'Gray', 'Green', 'More than 1 color', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    prediction_color = color_classes[int(result_color)]
    
    #check the pattern    
    pattern_classes = ['Floral', 'Graphics', 'Plaid', 'Solid', 'Spotted', 'Striped']    
    prediction_pattern = pattern_classes[int(result_pattern)]       

    prediction_for_color_pattern =  str(prediction_color) + ", "  + str(prediction_pattern) + " " +  str(acsessories_class) 
    custom_bbox(str(path),str(prediction_for_color_pattern) ,xmin ,  ymin, xmax, ymax, im_width, im_height)
    
"""
    Main function for reading the json data
    param: data json 
"""
def read_json_data(data):
    acsessories_list = ["b'backpack", "b'umbrella", "b'book", "b'cell phone", "b'tie", "b'suitcase", "b'handbag", "b'baseball glove", "b'tennis racket", "b'laptop" ]
    for element in data:
        current_image = element
        im_dictionary = data[str(current_image)]
        im_width = im_dictionary['width']
        im_height = im_dictionary['height']
        file_path = im_dictionary['file_path']
        box = im_dictionary['boxes']
        for index in range(len(box['classes'])):
            if(box['classes'][index] == "b'person'"):
                #take the bounding box on the image
                xmin = box['xmin'][index]
                ymin = box['ymin'][index]
                xmax = box['xmax'][index]
                ymax = box['ymax'][index]
                scores = box['scores'][index]
                #train with your own classifiers
                crop_image(str(file_path),xmin, ymin, xmax, ymax, im_width, im_height)
            elif(box['classes'][index] in acsessories_list):
                #take the bounding box on the image
                print(box['classes'][index])
                xmin = box['xmin'][index]
                ymin = box['ymin'][index]
                xmax = box['xmax'][index]
                ymax = box['ymax'][index]
                scores = box['scores'][index]
                #train with your own classifiers
                acsessory_pattern_color(str(box['classes'][index][2:]), str(file_path),xmin, ymin, xmax, ymax, im_width, im_height)
dire = "/kaggle/input/fashion-t/FASHION_AI/chris hemsworth/"
res = infer(dire,512)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from six import string_types
from six.moves import range
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

def _validate_label_map(label_map):
  """Checks if a label map is valid.

  Args:
    label_map: StringIntLabelMap to validate.

  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
        raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
        raise ValueError('Label map id 0 is reserved for the background label')


def load_labelmap_copy(path):
  """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.io.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
        text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
        label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map

### THESE LINES ARE OBSTRUCTING THE WORKING OF THE TWO MODELS SIMULTANEOUSLY
# from tensorflow.python.framework import ops
# ops.reset_default_graph()

%cd /kaggle/working/models/research/object_detection/

# %tensorflow_version 1.x
%matplotlib inline
#ALL LIBRARIES IMPORTED IN HERE
from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet
#make a prediction from your test
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv
from flask import Flask, current_app
import json
# from google.colab.patches import cv2_imshow


import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
############################################################################################
from object_detection.utils import ops as utils_ops
"*********************************************************************************"
#map imports
from utils import label_map_util
from utils import visualization_utils as vis_util
"*********************************************************************************"
# What model to download.
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

#number of classes in the COCO dataset
NUM_CLASSES = 90
"*********************************************************************************"
#set the model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
"********************************************************************************"
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
"*********************************************************************************"
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
label_map = load_labelmap_copy(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
"**********************************************************************************"
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
"**********************************************************************************"
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.compat.v1.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks( detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        #small test
        print( detection_masks_reframed )
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
"**************************************************************"
def predict_image(img_path_list):
    #initialize the dictionary
    image_json_dict = dict()
    #just to label images
    count = 0
    for image_path in img_path_list:
        print(image_path)
        image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
      # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],
                                                           category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)

    #             plt.figure(figsize=IMAGE_SIZE).suptitle(image_path)
        plt.imshow(image_np)
        plt.show()
    #         return output
        boxes = output_dict['detection_boxes']
        #initialize the lists here for bounding boxes not real one playing with format
        image_dict = dict()
        xmin_list = list()
        xmax_list = list()
        ymin_list = list()
        ymax_list = list()
        class_list = list()
        score_list = list()
        box_dict = dict()
        prediction_dict = dict()
        acsessory_list = list()
        clothing_list = list()
        #iterating over possible boxes
        for i in range(min(20, boxes.shape[0])):
            if output_dict['detection_scores'] is None or output_dict['detection_scores'][i] > 0.5:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box
                im_width, im_height = image.size
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax *im_height)
                print("for image: " + str(i) + ": " + str(left) +" , " + str(right) +" , " + str(output_dict['detection_classes'][i]) )
                objects = []
                for index, value in enumerate(output_dict['detection_classes']):
                    object_dict = {}
                    threshold = 0.5
                    if output_dict['detection_scores'][index] > threshold:
                        object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                            output_dict['detection_scores'][index]
                        objects.append(object_dict)
                #obtaining classes
                print(str(list(objects[i].keys())[0])) 
                #obtaining class scores
                print(float(list(objects[i].items())[0][1]))
                #gives the label of the detection you can get the score by .values()
                #add your boxes for constructing a json 
                class_list.append(str(list(objects[i].keys())[0]))
                xmin_list.append(int(left))
                ymin_list.append(int(top))
                xmax_list.append(int(right))
                ymax_list.append(int(bottom))
                score_list.append(float(list(objects[i].items())[0][1]))
                #I decided doing the saving here
                clothing = ""
                acsessory = ""
                #load image and pattern models here 
                im_width, im_height = image.size
                if(str(list(objects[i].keys())[0]) == "b'person'"):
                    clothing = crop_image(str(image_path),int(left), int(top), int(right), int(bottom), im_width, im_height)
                    acsessories_list = ["b'backpack", "b'umbrella", "b'book", "b'cell phone", "b'tie", "b'suitcase", "b'handbag", "b'baseball glove", "b'tennis racket", "b'laptop" ]
                if(str(list(objects[i].keys())[0]) in acsessories_list):
                    acsessory = acsessory_pattern_color(str(list(objects[i].keys())[0]), str(image_path),int(left), int(top),  int(right) ,int(bottom), im_width, im_height)
                #End of my experiment
                clothing_list.append(str(clothing))
                acsessory_list.append(str(acsessory))
        prediction_dict = {'acsessories': acsessory_list, 'clothes': clothing_list}
        #WORKING PLACE  
        box_dict = {'classes': class_list, 'xmin': xmin_list, 'ymin':  ymin_list, 'xmax': xmax_list, 'ymax':  ymax_list, 'scores' : score_list }  
        image_dict['boxes'] = box_dict
        image_dict['prediction result'] = prediction_dict
        im_width, im_height = image.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax *im_height)
        image_dict['file_path'] = str(image_path)
        image_dict['width'] = int(im_width)
        image_dict['height'] = int(im_height)
        ymin, xmin, ymax, xmax = box
        image_name = image_path.split('/')[-1].split('.')[-2]
        image_json_dict[image_name] = image_dict
        count+=1
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
    return image_json_dict
PATH_TO_TEST_IMAGES_DIR =  "/kaggle/input/fashion-t/FASHION_AI/chris hemsworth/"
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,'Image_{}.jpg'.format(i)) for i in range(1,6)]# int(len(PATH_TO_TEST_IMAGES_DIR))) ] #runs prediction on the every image in the directory
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
json_dict = predict_image(TEST_IMAGE_PATHS)
img = plt.imread("/kaggle/input/fashion-t/FASHION_AI/chris hemsworth/Image_14.jpg")
plt.imshow(img)
img = cv2.resize(img,(512,512))
imgs = []
for mask_idx in range(res[1][0]['masks'].shape[2]):
    plt.figure()
    a = img.copy()
    a[res[1][0]['masks'][:,:,mask_idx]==0]=0
#     plt.imshow(res[1][0]['masks'][:,:,mask_idx])
    imgs.append(a)
    plt.imshow(a)
    plt.show()
# res[1][0]['masks']
"**************************************************************"
def predict_image_arr(img_array):
    #initialize the dictionary
    image_json_dict = dict()
    #just to label images
    count = 0
    for image_np in img_array:
      # the array based representation of the image will be used later in order to prepare the result image with boxes and labels on it.
#         image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
      # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],
                                                           category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)

        plt.figure(figsize=IMAGE_SIZE).suptitle(count)
        plt.imshow(image_np)
        plt.show()
    return output_dict
#     #         return output
#         boxes = output_dict['detection_boxes']
#         #initialize the lists here for bounding boxes not real one playing with format
#         image_dict = dict()
#         xmin_list = list()
#         xmax_list = list()
#         ymin_list = list()
#         ymax_list = list()
#         class_list = list()
#         score_list = list()
#         box_dict = dict()
#         prediction_dict = dict()
#         acsessory_list = list()
#         clothing_list = list()
#         #iterating over possible boxes
#         m = min(20, boxes.shape[0])
#         print(m)
#         for i in range(m):
#             if output_dict['detection_scores'] is not None or output_dict['detection_scores'][i] > 0.5:
#                 box = tuple(boxes[i].tolist())
#                 ymin, xmin, ymax, xmax = box
#                 im_width, im_height = image_np.shape[:2]
#                 (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax *im_height)
#                 print("for image: " + str(i) + ": " + str(left) +" , " + str(right) +" , " + str(output_dict['detection_classes'][i]) )
#                 objects = []
#                 for index, value in enumerate(output_dict['detection_classes']):
#                     object_dict = {}
#                     threshold = 0.5
#                     if output_dict['detection_scores'][index] > threshold:
#                         object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
#                             output_dict['detection_scores'][index]
#                         objects.append(object_dict)
#                 #obtaining classes
#                 print(i)
# #                 print(str(list(objects[i].keys())[0])) 
#                 #obtaining class scores
# #                 print(float(list(objects[i].items())[0][1]))
#                 #gives the label of the detection you can get the score by .values()
#                 #add your boxes for constructing a json 
#                 class_list.append(str(list(objects[i].keys())[0]))
#                 xmin_list.append(int(left))
#                 ymin_list.append(int(top))
#                 xmax_list.append(int(right))
#                 ymax_list.append(int(bottom))
#                 score_list.append(float(list(objects[i].items())[0][1]))
#                 #I decided doing the saving here
#                 clothing = ""
#                 acsessory = ""
#                 #load image and pattern models here 
#                 im_width, im_height = image_np.shape[:2]
#                 if(str(list(objects[i].keys())[0]) == "b'person'"):
#                     clothing = crop_image(str(image_path),int(left), int(top), int(right), int(bottom), im_width, im_height)
#                     acsessories_list = ["b'backpack", "b'umbrella", "b'book", "b'cell phone", "b'tie", "b'suitcase", "b'handbag", "b'baseball glove", "b'tennis racket", "b'laptop" ]
#                 if(str(list(objects[i].keys())[0]) in acsessories_list):
#                     acsessory = acsessory_pattern_color(str(list(objects[i].keys())[0]), str(image_path),int(left), int(top),  int(right) ,int(bottom), im_width, im_height)
#                 #End of my experiment
#                 clothing_list.append(str(clothing))
#                 acsessory_list.append(str(acsessory))
#         prediction_dict = {'acsessories': acsessory_list, 'clothes': clothing_list}
#         #WORKING PLACE  
#         box_dict = {'classes': class_list, 'xmin': xmin_list, 'ymin':  ymin_list, 'xmax': xmax_list, 'ymax':  ymax_list, 'scores' : score_list }  
#         image_dict['boxes'] = box_dict
#         image_dict['prediction result'] = prediction_dict
#         im_width, im_height = image_np.shape[:2]
#         (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax *im_height)
#         image_dict['file_path'] = str(image_path)
#         image_dict['width'] = int(im_width)
#         image_dict['height'] = int(im_height)
#         ymin, xmin, ymax, xmax = box
#         image_json_dict[count] = image_dict
#         count+=1
#         print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
#     return image_json_dict
d = predict_image_arr(imgs)
d
