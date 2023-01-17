# training chart showing mAP score and iteration details

from IPython.display import Image

Image("../input/wheat-yolov4-training-results/chart_wheat_608.png")
# console output of the detection with default iou threshold

Image("../input/wheat-yolov4-training-results/detect_map_wheat.JPG")
# console output of the detection with custom iou threshold

Image("../input/wheat-yolov4-training-results/detect_map_wheat_thres.JPG")
Image("../input/wheat-yolov4-training-results/2fd875eaa.jpg")
Image("../input/wheat-yolov4-training-results/348a992bb.jpg")
Image("../input/wheat-yolov4-training-results/51b3e36ab.jpg")
Image("../input/wheat-yolov4-training-results/51f1be19e.jpg")
Image("../input/wheat-yolov4-training-results/53f253011.jpg")
Image("../input/wheat-yolov4-training-results/796707dd7.jpg")
Image("../input/wheat-yolov4-training-results/aac893a91.jpg")
Image("../input/wheat-yolov4-training-results/cb8d261a3.jpg")
Image("../input/wheat-yolov4-training-results/cc3532ff6.jpg")
Image("../input/wheat-yolov4-training-results/f5a1f0358.jpg")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/working/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# import required libraries

import io

import time

import math

import cv2, colorsys

from PIL import Image, ImageEnhance, ImageFilter

import imgaug.augmenters as iaa

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb



from functools import wraps, reduce



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import load_model

from tensorflow.lite.python import interpreter as interpreter_wrapper

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, MaxPooling2D, BatchNormalization, Activation, UpSampling2D, ZeroPadding2D

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.regularizers import l2



print("TensorFlow version is: {}".format(tf.__version__))

print("Eager execution is: {}".format(tf.executing_eagerly()))

print("Keras version is: {}".format(tf.keras.__version__))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!python ../input/yolo2keraswbf/keras-YOLOv3-model-set/tools/evaluation/validate_yolo.py --model_path=../input/yolo2keraswbf/keras-YOLOv3-model-set/weights/wheat_yolov4.h5 --image_file=../input/global-wheat-detection/test/ --anchors_path=../input/yolo2keraswbf/keras-YOLOv3-model-set/configs/yolo4_anchors.txt --classes_path=../input/yolo2kerastta/keras-YOLOv3-model-set/configs/wheat_classes.txt --model_image_size=1024x1024
print("+++++++++++++++++++++++++++++++++ Completed +++++++++++++++++++++++++++++++++++++++++++")