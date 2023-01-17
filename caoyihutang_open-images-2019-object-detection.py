

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''



# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input

from keras.utils.data_utils import GeneratorEnqueuer

import matplotlib.pyplot as plt

import math, os
image_path = '/kaggle/input/open-images-2019-object-detection/'



batch_size = 100

img_generator = ImageDataGenerator().flow_from_directory(image_path, shuffle=False, target_size = (416,416), batch_size = batch_size)

n_rounds = math.ceil(img_generator.samples / img_generator.batch_size)

filenames = img_generator.filenames



img_generator = GeneratorEnqueuer(img_generator)

img_generator.start()

img_generator = img_generator.get()

print(img_generator)
class_descriptions_boxable = pd.read_csv('/kaggle/input/open-images-classes/class-descriptions-boxable.csv',header=None)

rev = class_descriptions_boxable.set_index(1).T.to_dict('list')
!cp -r ../input/imageaireport/imageai/imageai imageai
from imageai.Detection import ObjectDetection

model_weight_path = "../input/modelyolov3/model_yolov3/model_yolov3.h5"



execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()

detector.setModelPath(model_weight_path)

detector.loadModel()
for i in range(n_rounds):

    batch = next(img_generator)

    for j, prediction in enumerate(batch):

        image = filenames[i * batch_size + j]

        detections = detector.detectObjectsFromImage(input_image=image_path+image, output_image_path="image_with_box.png", minimum_percentage_probability = 75)        

        pred_str = ""

        labels = ""

        for eachObject in detections:

            if eachObject["name"].capitalize() in rev:

                pred_str += rev[eachObject["name"].capitalize()][0] + " " + str(float(eachObject["percentage_probability"])/100) + " 0.1 0.1 0.9 0.9"

                pred_str += " "

                labels += eachObject['name'] + ", " + str(round(float(eachObject['percentage_probability'])/100, 1)) 

                labels += " | "

        if labels != "":

            plt.figure(figsize=(12,12))

            plt.imshow(plt.imread("image_with_box.png"))

            plt.show()



            print ("Labels Detected: ")

            print (labels)

            print ()

            print ("Prediction String: ")

            print (pred_str)



    if i == 10:

        break