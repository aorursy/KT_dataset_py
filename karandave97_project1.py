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
import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np 

import math, os

from keras.applications.inception_v3 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from keras.utils.data_utils import GeneratorEnqueuer







#image_path ="../input/google-ai-open-images-object-detection-track/test/"

image_path="../input/test/"

batch_size = 100

img_generator = ImageDataGenerator().flow_from_directory(image_path, shuffle=False, batch_size = batch_size)                                                                                      

#calculating size of epoch                                                                                         

n_rounds = math.ceil(img_generator.samples / img_generator.batch_size)  # size of an epoch



filenames = img_generator.filenames

img_generator = GeneratorEnqueuer(img_generator)

img_generator.start()

img_generator = img_generator.get()

from imageai.Detection import ObjectDetection

model_weight_path = "../input/imageairepo/imageai/resnet50_v2.0.1.h5"



execution_path = os.getcwd()

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()

detector.setModelPath(model_weight_path)

detector.loadModel()