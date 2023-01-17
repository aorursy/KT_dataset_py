# Let's see the test data.

!ls ../input
# Python lib

import pandas as pd

import numpy as np

from tqdm import tqdm_notebook



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf



import os

import sys
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()
test_filename = os.listdir('../input/test')

test_filename[:5]
# Show one image

def show_image_by_index(i):

    sample_image = plt.imread(f'../input/test/{test_filename[i]}')

    plt.imshow(sample_image)



def show_image_by_filename(filename):

    sample_image = plt.imread(filename)

    plt.imshow(sample_image)

    

show_image_by_index(1)
# Test Image AI

# First install the python lib

!pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
!ls -l
from imageai.Detection import ObjectDetection
execution_path = os.getcwd()
%%time

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()

detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5") )

detector.loadModel()
%%time

detections = detector.detectObjectsFromImage(input_image=os.path.join('../input/test' , 

                                                                      test_filename[1]), 

                                             output_image_path=os.path.join(execution_path , "result.jpg"),

#                                              output_type = 'array',

                                             extract_detected_objects = False)

for eachObject in detections:

    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )



# show the result

show_image_by_filename('./result.jpg')
detections
def format_prediction_string(image_id, result):

    prediction_strings = []

    

    for i in range(len(result['percentage_probability'])):

        class_name = result['name'][i].decode("utf-8")

        boxes = result['detection_boxes'][i]

        score = result['percentage_probability'][i]

        

        prediction_strings.append(

            f"{class_name} {score} " + " ".join(map(str, boxes))

        )

        

    prediction_string = " ".join(prediction_strings)



    return {

        "ImageID": image_id,

        "PredictionString": prediction_string

    }