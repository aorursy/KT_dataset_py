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
#Download the latest release 2.0.2 of ImageAI

!pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
#Download YOLO v3 model

!wget "https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5"
#check for downloaded model

print(os.listdir(os.getcwd()))
#download the classes data from official competetion page

df_classes = pd.read_csv("https://storage.googleapis.com/openimages/challenge_2018/challenge-2018-class-descriptions-500.csv", header=None)

df_classes.columns = ['imageClassId', 'imageClass']

df_classes.shape #check if all the classes are fetched

dict_classes = dict(zip(df_classes.imageClassId, df_classes.imageClass))  #convert the df into dictionary 

len(dict_classes)
#converting all classnames to lowercase

image_classes = {}

for k,v in dict_classes.items():

    image_classes[v.lower()] = k
from imageai.Detection import ObjectDetection

import os



model = ObjectDetection()

execution_path = os.getcwd()

model.setModelTypeAsYOLOv3()

model.setModelPath('/kaggle/working/yolo.h5') 

model.loadModel()
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import preprocess_input

from keras.utils.data_utils import GeneratorEnqueuer

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np 

import math, os

from PIL import  Image

from progressbar import progressbar as pbar

import glob
image_path = "../input/google-ai-open-images-object-detection-track/test/challenge2018_test/"

test_images = os.listdir(image_path)

#Complete processing and object detection with coordinates of each image takes 0.4 secs, 

#so, divide the 99,999 test images into batches if you want

test_images_batch = test_images[:2] #I am only taking 2 images in the batch here to reduce compile time, as i have already predicted all images earlier by dividing them into bacthes

df_output = pd.DataFrame(columns=['ImageId', 'PredictionString']) #Create a dataframe and add results to it

#dummy = 1

for image in pbar(test_images_batch):

    #print('Image working on is: ', image,' and dummy variable value is: ',dummy)    

    detections = model.detectObjectsFromImage(input_image=image_path+image, output_image_path="image_with_box.png", minimum_percentage_probability = 65)  

    #print(detections)

    ImageId = str(image).split('.')[0]

    im = Image.open(image_path+image)

    image_width, image_height = im.size

    pred_str = ""

    labels = ""

    for eachObject in detections:                   

        x1,y1,x2,y2 = eachObject["box_points"]

        box_pts = str(round(x1/image_width,2))+" "+str(round(y1/image_height,2))+" "+str(round(x2/image_width,2))+" "+str(round(y2/image_height,2))

        if eachObject["name"] in image_classes:

            pred_str += image_classes[eachObject["name"]] + " " + str(round(float(eachObject["percentage_probability"])/100,2)) +" "+ box_pts + " "

        else:

            #if the detected class is not present in classes given, there wont be any classid , then istead of classid, put the classname detected

            pred_str += eachObject["name"] + " " + str(round(float(eachObject["percentage_probability"])/100,2)) +" "+ box_pts + " "

        labels += eachObject['name'] + ", " + str(round(float(eachObject['percentage_probability'])/100, 1)) 

        labels += " | "    

            

    df_output = df_output.append({'ImageId': ImageId, 'PredictionString': pred_str}, ignore_index=True)



print("Completed predictions of all test images")

df_output.set_index('ImageId', inplace=True)

#df_output.to_csv('predictions.csv') #uncomment this before running to get predcition results for the batch in predictions.csv
os.chdir("../input/resultsgooglechallengetoupload/googleopenimageschallengeresults/GoogleOpenImagesChallengeResults/")

results = pd.DataFrame([])

 

for counter, file in enumerate(glob.glob("*")):

    namedf = pd.read_csv(file)

    results = results.append(namedf)

    

results.set_index('ImageId', inplace=True)

os.chdir('/kaggle/working/') 

results.to_csv('predictions_combinedfile.csv')