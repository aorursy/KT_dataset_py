!pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl
import pandas as pd

import numpy as np

from bq_helper import BigQueryHelper

from matplotlib import pyplot as plt

import skimage

import skimage.util

import cv2

from urllib.request import urlretrieve

from imageai.Prediction import ImagePrediction

from tqdm import tqdm_notebook



%matplotlib inline
open_images = BigQueryHelper(active_project="bigquery-public-data", dataset_name="open_images")

open_images.list_tables()
query = """

            SELECT *

            FROM `bigquery-public-data.open_images.images` 

            WHERE image_id IN 

            (SELECT image_id

            FROM `bigquery-public-data.open_images.labels`

            WHERE label_name = '/m/01g317' and confidence > 0.8

            LIMIT 10000

            )

            LIMIT 100

            

        """

urls = open_images.query_to_pandas_safe(query, max_gb_scanned=10)

urls
!mkdir images
# get only 300k quality images

samples = range(10)

image_url_samples_with_person = urls["thumbnail_300k_url"].iloc[samples].tolist()



# download images and load it in

list_images_samples = []

for img_file in image_url_samples_with_person:

    img_data = skimage.io.imread(img_file)

    np_img = np.array(img_data)

    np_img = skimage.transform.resize(np_img, (256, 256))

    

    # convert grayscale images to rgb by just tinting the three dimensions

    if len(np_img.shape) == 2:

        np_img = skimage.color.gray2rgb(np_img)

    

    list_images_samples.append(np_img)
plt.figure(figsize=(20,20))

plt.imshow(skimage.util.montage(np.array(list_images_samples[:9]), multichannel=True), cmap='gray')

plt.title("Sample images")

plt.show()
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
from imageai.Detection import ObjectDetection



detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()

detector.setModelPath("yolo.h5")

detector.loadModel()
plt.imshow((list_images_samples[0] * 255).astype(np.int32))
input_img = (list_images_samples[0] * 255).astype(np.int32)

detections = detector.detectObjectsFromImage(input_img, input_type='array', 

                                             minimum_percentage_probability=50, output_type='array')
plt.imshow(detections[0])
list_image_detections = []

list_dict_results = []



for np_img in tqdm_notebook(list_images_samples):

    input_img = (np_img * 255).astype(np.int32)

    detections = detector.detectObjectsFromImage(input_img, input_type='array', 

                                                 minimum_percentage_probability=50, output_type='array')

    

    list_image_detections.append(detections[0])

    list_dict_results.append(detections[1])
plt.figure(figsize=(20,20))

plt.imshow(skimage.util.montage(np.array(list_image_detections[:9]), multichannel=True), cmap='gray')

plt.title("Objects detected in images using YOLO")

plt.show()
list_dict_results[0]
np_img_montage = skimage.util.montage(np.array(list_images_samples[:9]), multichannel=True)
input_img = (np_img_montage * 255).astype(np.int32)

detections = detector.detectObjectsFromImage(input_img, input_type='array', 

                                             minimum_percentage_probability=50, output_type='array')
plt.figure(figsize=(20, 20))

plt.imshow(detections[0])

plt.title("Objects detected if all images are stitched and processed in one go")
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5
from imageai.Detection import ObjectDetection



detector = ObjectDetection()

detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath("yolo-tiny.h5")

detector.loadModel()
list_image_detections = []

list_dict_results = []



for np_img in tqdm_notebook(list_images_samples):

    input_img = (np_img * 255).astype(np.int32)

    detections = detector.detectObjectsFromImage(input_img, input_type='array', 

                                                 minimum_percentage_probability=50, output_type='array')

    

    list_image_detections.append(detections[0])

    list_dict_results.append(detections[1])
plt.figure(figsize=(20,20))

plt.imshow(skimage.util.montage(np.array(list_image_detections[:9]), multichannel=True), cmap='gray')

plt.title("Detected objects using Tiny YOLO")

plt.show()
# get only 300k quality images

samples = range(10, 20)

image_url_samples_with_person = urls["thumbnail_300k_url"].iloc[samples].tolist()



# download images and load it in

list_images_samples = []

for img_file in image_url_samples_with_person:

    img_data = skimage.io.imread(img_file)

    np_img = np.array(img_data)

    np_img = skimage.transform.resize(np_img, (256, 256))

    

    # convert grayscale images to rgb by just tinting the three dimensions

    if len(np_img.shape) == 2:

        np_img = skimage.color.gray2rgb(np_img)

    

    list_images_samples.append(np_img)
plt.figure(figsize=(20,20))

plt.imshow(skimage.util.montage(np.array(list_images_samples[:9]), multichannel=True), cmap='gray')

plt.title("Sample Images")

plt.show()
from imageai.Detection import ObjectDetection



detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()

detector.setModelPath("yolo.h5")

detector.loadModel()
list_image_detections = []

list_dict_results = []



for np_img in tqdm_notebook(list_images_samples):

    input_img = (np_img * 255).astype(np.int32)

    detections = detector.detectObjectsFromImage(input_img, input_type='array', 

                                                 minimum_percentage_probability=50, output_type='array')

    

    list_image_detections.append(detections[0])

    list_dict_results.append(detections[1])
plt.figure(figsize=(20,20))

plt.imshow(skimage.util.montage(np.array(list_image_detections[:9]), multichannel=True), cmap='gray')

plt.title("Objects detected in images using YOLO")

plt.show()
list_dict_results[5]
# get only 300k quality images

samples = range(30, 40)

image_url_samples_with_person = urls["thumbnail_300k_url"].iloc[samples].tolist()



# download images and load it in

list_images_samples = []

for img_file in image_url_samples_with_person:

    img_data = skimage.io.imread(img_file)

    np_img = np.array(img_data)

    np_img = skimage.transform.resize(np_img, (256, 256))

    

    # convert grayscale images to rgb by just tinting the three dimensions

    if len(np_img.shape) == 2:

        np_img = skimage.color.gray2rgb(np_img)

    

    list_images_samples.append(np_img)
plt.figure(figsize=(20,20))

plt.imshow(skimage.util.montage(np.array(list_images_samples[:9]), multichannel=True), cmap='gray')

plt.title("Sample Images")

plt.show()
list_image_detections = []

list_dict_results = []



for np_img in tqdm_notebook(list_images_samples):

    input_img = (np_img * 255).astype(np.int32)

    detections = detector.detectObjectsFromImage(input_img, input_type='array', 

                                                 minimum_percentage_probability=50, output_type='array')

    

    list_image_detections.append(detections[0])

    list_dict_results.append(detections[1])
plt.figure(figsize=(20,20))

plt.imshow(skimage.util.montage(np.array(list_image_detections[:9]), multichannel=True), cmap='gray')

plt.title("Objects detected in images using YOLO")

plt.show()