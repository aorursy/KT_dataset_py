!pip install keras_segmentation
!pip install pyspark
import cv2

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





### Scikit learn libraries ### 

from skimage.color import rgb2gray

from skimage.feature import blob_dog

from keras.models import model_from_json





### Pyspark libraries ###

import os,gc

import shutil

import pyspark

import itertools

from operator import add

from pyspark.sql.types import *

from pyspark import SparkContext

from pyspark.sql import DataFrame

from pyspark.sql.functions import udf

from pyspark.ml.image import ImageSchema
file = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")  # This file contains bbox co-ordinates of wheat spikes for each image.



# Train csv has multiple rows for each image depending upon number of wheat spikes in each image.

ids = [file.iloc[i][0] for i in range(file.shape[0])] 



# Factor by which we will resize each image as each raw image has size 1024*1024, which will be too large for network to train on to.

factor = 2; size = 512; batch_size = 500; x_axis_gaps = 5; coords_dist_th = 15; N = 3100; N_50 = N+50
sample = os.listdir("/kaggle/input/global-wheat-detection/train")[0:3]



for img_name in sample:

    image = cv2.imread('/kaggle/input/global-wheat-detection/train/'+ img_name)

    

    # Get image ids for each image to access the segmentation co-ordinates

    image_ids = [l for l,val in enumerate(ids) if val == str(img_name.split(".")[0])]

    list_of_coords = [[int(float(val)) for val in file.iloc[v][3][1:len(file.iloc[v][3])-1].split(",")] for v in image_ids] 

            

    for l in list_of_coords:        

        image_annotated = cv2.rectangle(image,(int(l[0]),int(l[1])),(int((l[0]+l[2])),int((l[1]+l[3]))),(255,0,0),4)



    fig, axes = plt.subplots(1, 2, figsize=(12, 6))#, sharex=True, sharey=True)

    ax = axes.ravel()

    image = cv2.imread('/kaggle/input/global-wheat-detection/train/'+ img_name)    

    ax[0].imshow(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))

    ax[1].imshow(cv2.cvtColor(image_annotated,cv2.COLOR_RGB2BGR))

        

    plt.show()
if not os.path.exists('train_annotated'):

        os.makedirs('train_annotated')

        

def data_prep(img_name):

    '''

    img_name : input image name 

    

    '''

    

    # Initialising a balck n image of zeros which we will later manipulate using annotation information.

    binary_img = np.zeros((size, size, 3))

    

    image = cv2.imread('spark_temp/'+ img_name.split(".")[0] + ".png") #reading actual image ".png")

    

    # Get image ids for each image to access the segmentation co-ordinates

    image_ids = [l for l,val in enumerate(ids) if val == str(img_name.split(".")[0])]

    list_of_coords = [[int(float(val)) for val in file.iloc[v][3][1:len(file.iloc[v][3])-1].split(",")] for v in image_ids] 

    if len(list_of_coords) >= 0:#>1

        for l in list_of_coords:

            # Getting segment for wheat spike, co-ordinates adjusted 

            cropped = image[int(l[1]/factor):int((l[1]+l[3])/factor), int(l[0]/factor):int((l[0]+l[2])/factor)]   



            for i,x in enumerate(range(int(l[1]/factor),int((l[1]+l[3])/factor))):

                for j,y in enumerate(range(int(l[0]/factor),int((l[0]+l[2])/factor))):

                    x = min(image.shape[0]-1,x)

                    y = min(image.shape[0]-1,y)

                    try:

                        binary_img[x,y][0] = (image[x,y][0]/np.max(cropped[:,:,0]))*50

                        # normalising the image area where wheat spike exists



                    except:pass



        binary_img[binary_img < np.mean(binary_img)] = 0

        binary_img[binary_img >= np.mean(binary_img)] = 1

#         binary_img = cv2.resize(binary_img, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite("train_annotated/" + str(img_name.split(".")[0]) + ".png",binary_img)
def run_spark():   

    sc=SparkContext(master="local[15]") # number[15] can be changed based on your system specification.

#     print(sc.binaryFiles("spark_temp/*.png"))

    image_df = sc.binaryFiles("spark_temp") # Reading images into Spark context

    

#     pyspark.sql.udf.UDFRegistration.register(name="data_prep", f = data_prep, returnType=StringType()) #registering UDF 

    # such that Spark context recongises the function used for data preparation.

    

    job = [data_prep(x[0].split("/")[-1]) for idx,x in enumerate(image_df.take(batch_size))]

    shutil.rmtree('spark_temp') #remove the folder once 'one batch' is complete to avoid Spark remembering 

    # indices for images it is done with.

    sc.stop() # stop the spark context else Spark will have unnecessary information cached whcih we do not require anymore.

    gc.collect() # remove any other cache which memory might have been holding.

    
input_list = os.listdir("/kaggle/input/global-wheat-detection/train/")[0:N]

input_seq = sorted(list(set([val for val in range(0, len(input_list),batch_size)]+ [len(input_list)])))



def image_transfer(value):

    if not os.path.exists('spark_temp'):

        os.makedirs('spark_temp')

    image = cv2.imread('/kaggle/input/global-wheat-detection/train/'+value)

    cv2.imwrite("spark_temp/" + value.split(".")[0]+".png",cv2.resize(image,(size,size)))#cv2.resize(image,(size,size))

    

    

for val in range(len(input_seq)-1):

    img_batch = input_list[input_seq[val]:input_seq[val]+batch_size]

    [image_transfer(val) for val in img_batch]

  

    run_spark()
if not os.path.exists('train_temp'):

        os.makedirs('train_temp')

def image_transfer_(value):      

    cv2.imwrite("train_temp/" + value.split(".")[0]+".png",cv2.resize(cv2.imread('/kaggle/input/global-wheat-detection/train/'+value.split(".")[0]+".jpg"),(size,size)))

    

first_3000 = os.listdir("train_annotated")

task = [image_transfer_(val) for val in first_3000]
error_list = [val if np.max(cv2.imread('train_annotated/'+val)) > 1 else None for val in os.listdir("train_annotated")]

error_list = [val for val in error_list if val != None]

print(error_list)

[os.remove("train_temp/"+val) for val in error_list]

[os.remove("train_annotated/"+val) for val in error_list]
from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=2 ,  input_height=size, input_width=size)

model.train(

    train_images =  "train_temp/",#train

    train_annotations = "train_annotated/",n_classes = 2,epochs=10,steps_per_epoch=5,#annotations_prepped_train_v3

)
 #saving model to disk



from keras.models import model_from_json

model_json = model.to_json()



with open("model.json", "w") as json_file:

    json_file.write(model_json)

model.save_weights("model.h5")

if not os.path.exists('test_'):

        os.makedirs('test_')

def test_image_transfer(value):      

    cv2.imwrite("test_/" + value.split(".")[0]+".png",cv2.resize(cv2.imread('/kaggle/input/global-wheat-detection/test/'+value.split(".")[0]+".jpg"),(size,size)))

    

test_imgs = os.listdir("/kaggle/input/global-wheat-detection/test")

task = [test_image_transfer(val) for val in test_imgs]





if not os.path.exists('train_0'):

        os.makedirs('train_0')

def train_image_transfer(value):      

    cv2.imwrite("train_0/" + value.split(".")[0]+".png",cv2.resize(cv2.imread('/kaggle/input/global-wheat-detection/train/'+value.split(".")[0]+".jpg"),(size,size)))

    

train_imgs = os.listdir("/kaggle/input/global-wheat-detection/train/")[N:N_50]

task = [train_image_transfer(val) for val in train_imgs]
import warnings,os

warnings.simplefilter("ignore")

submission = {"image_id" : [], "width" : [], "height" : [],"bbox" : [], "Prob":[]}



for idx,img_name in enumerate(os.listdir("test_")[0:5]):

    if not os.path.exists('segmentation'):

        os.makedirs('segmentation')

    if not os.path.exists('Output'):

        os.makedirs('Output')

        

    

    out = model.predict_segmentation(

    inp="test_/"+ img_name,

    out_fname="segmentation/" + img_name.split(".")[0] + ".png" 

    )  

    

    out_image = cv2.imread('segmentation/'+ img_name.split(".")[0] + ".png")

    _image = cv2.imread('test_/'+ img_name.split(".")[0] + ".png")

#     print(out_image)

#     out_image = cv2.resize(out_image,(512,512))



    ## Step 4:Blob detection

    image_gray = rgb2gray(out_image)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.10)

    

    try:

        blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)

        submission["image_id"].append(img_name.split(".")[0])

        submission["width"].append(size); submission["height"].append(size)

        submission["bbox"].append([str([int(blob[0]), int(blob[1]),int(blob[2]), int(blob[2])]) for blob in blobs_dog])

        submission["Prob"].append(.50)

    except:

        submission["image_id"].append(img_name.split(".")[0])

        submission["width"].append(size); submission["height"].append(size)

        submission["bbox"].append([])

        submission["Prob"].append(1)



    

    blob_list_sorted = sorted([[int(blob[0]), int(blob[1]),int(blob[2])] for blob in blobs_dog])

    len_val = len(blob_list_sorted)

    for i,val in enumerate(range(0,len(blob_list_sorted))):

        for j in range(i+1,len(blob_list_sorted)-2):

            if max(i,j) < len_val and max([abs(np.diff(x)) for x in zip(blob_list_sorted[j][0:2], blob_list_sorted[i][0:2])]) < coords_dist_th:

                blob_list_sorted = blob_list_sorted+[[min(x) for x in zip(blob_list_sorted[j], blob_list_sorted[i])][0:2]+[blob_list_sorted[j][2]+blob_list_sorted[i][2]]]

                del blob_list_sorted[i]

                del blob_list_sorted[j]  

                len_val = len(blob_list_sorted)

   

    try:

        for blob in blobs_dog:                   

            y, x, r = blob

            Iou = cv2.rectangle(_image,(int(x),int(y)),(int((x+r)),int((y+r))),(255,0,0),2)         

        

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        ax = axes.ravel()



        ax[0].imshow(cv2.cvtColor(cv2.imread('test_/'+ img_name.split(".")[0] + ".png"),cv2.COLOR_RGB2BGR))

        ax[1].imshow(Iou)



        plt.show()

        cv2.imwrite("Output/" + img_name.split(".")[0] + ".png",Iou)

    except:pass

   
pd.DataFrame(submission).to_csv("submission.csv") 
warnings.simplefilter("ignore")



for idx,img_name in enumerate(os.listdir("train_0/")):

    if not os.path.exists('segmentation'):

        os.makedirs('segmentation')

    

    

    out = model.predict_segmentation(

    inp="train_0/"+ img_name,

    out_fname="segmentation/" + img_name.split(".")[0] + ".png" 

    )  

    

    out_image = cv2.imread('segmentation/'+ img_name.split(".")[0] + ".png")

    _image = cv2.imread('train_0/'+ img_name.split(".")[0] + ".png")





    ## Step 4:Blob detection

    image_gray = rgb2gray(out_image)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.10)

    

    blob_list_sorted = sorted([[int(blob[0]), int(blob[1]),int(blob[2])] for blob in blobs_dog])

    len_val = len(blob_list_sorted)

    for i,val in enumerate(range(0,len(blob_list_sorted))):

        for j in range(i+1,len(blob_list_sorted)-2):

            if max(i,j) < len_val and max([abs(np.diff(x)) for x in zip(blob_list_sorted[j][0:2], blob_list_sorted[i][0:2])]) < coords_dist_th:

                blob_list_sorted = blob_list_sorted+[[min(x) for x in zip(blob_list_sorted[j], blob_list_sorted[i])][0:2]+[blob_list_sorted[j][2]+blob_list_sorted[i][2]]]

                del blob_list_sorted[i]

                del blob_list_sorted[j]  

                len_val = len(blob_list_sorted)

   

    try:

        for blob in blobs_dog:                   

            y, x, r = blob

            Iou = cv2.rectangle(_image,(int(x),int(y)),(int((x+r)),int((y+r))),(255,0,0),2)         

        

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        ax = axes.ravel()



        ax[0].imshow(cv2.cvtColor(cv2.imread('train_0/'+ img_name.split(".")[0] + ".png"),cv2.COLOR_RGB2BGR))

        ax[1].imshow(Iou)



        plt.show()

        cv2.imwrite("Output/" + img_name.split(".")[0] + ".png",Iou)

    except:pass

   