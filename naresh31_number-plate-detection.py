# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/tensorflow-license-plate-detection-master/test_images"))



# Any results you write to the current directory are saved as output.
import numpy as np

import os

import sys

import tensorflow as tf

from matplotlib import pyplot as plt

from PIL import Image

import cv2

import pytesseract
def custom_plate():

    from custom_plate import allow_needed_values as anv 

    from custom_plate import do_image_conversion as dic
get_ipython().magic('matplotlib inline')

sys.path.append("..")
def label_map_util():

    from utils import label_map_util

    from utils import visualization_utils as vis_util
MODEL_NAME = 'numplate'

PATH_TO_CKPT = MODEL_NAME + '/graph-200000/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
detection_graph = tf.Graph()

with detection_graph.as_default():

  od_graph_def = tf.GraphDef()

  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:

    serialized_graph = fid.read()

    od_graph_def.ParseFromString(serialized_graph)

    tf.import_graph_def(od_graph_def, name='')