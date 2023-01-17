!cp -r ../input/imageai/imageai/imageai/ imageai
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input

from keras.utils.data_utils import GeneratorEnqueuer

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np 

import math, os

from imageai.Detection import ObjectDetection

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
image_path = "../input/sate-data/test_data_images/test_data_images/"

batch_size = 2703

img_generator = ImageDataGenerator().flow_from_directory(image_path, shuffle=False, batch_size = batch_size)

n_rounds = math.ceil(img_generator.samples / img_generator.batch_size)

filenames = img_generator.filenames



img_generator = GeneratorEnqueuer(img_generator)

img_generator.start()

img_generator = img_generator.get()
filenames[0]
execution_path = os.getcwd()

detector = ObjectDetection()

detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath( os.path.join(execution_path , "../input/yolo-tiny/yolo-tiny.h5"))

detector.loadModel()

#custom_objects = detector.CustomObjects(car=True)

outputfiles_name = {}

for i in range(2600,2703):

    class_name = ''

    probability = ''

    box_points = ''

    

    detections = detector.detectObjectsFromImage(input_image=image_path+filenames[i])

    

#     for eachObject in detections:

#         print(str(eachObject["name"]) + " : " + str(eachObject["percentage_probability"]))

#         print("--------------------------------")

    if len(detections) != 0 :

        class_name = '1 ' 

        probability = str(detections[0]['percentage_probability'] / 100) + ' '

        box_points = ' '.join([str(x) for x in detections[0]['box_points']])

    

    outputfiles_name[filenames[i][7:len(filenames[i])-4]+".txt"] = class_name+probability+box_points
import json

text_file = open("output.txt", "w")

text_file.write(json.dumps(outputfiles_name))

text_file.close()
outputfiles_name