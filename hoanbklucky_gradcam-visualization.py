# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from gradcam import GradCAM

#import initialization

!pip install -q /kaggle/input/imutils/imutils-0.5.3
print(os.listdir("/kaggle/input/test-images/"))
import matplotlib.pyplot as plt

image_path = "/kaggle/input/test-images/soccer_ball.jpg"

orig = plt.imread(image_path)

plt.imshow(orig)
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications import VGG16

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications import imagenet_utils

#from google.colab.patches import cv2_imshow

import numpy as np

import argparse

import cv2



# construct the argument parser and parse the arguments





# initialize the model to be VGG16

#Model = VGG16



# check to see if we are using ResNet

#if args["model"] == "resnet":

Model = ResNet50



# load the pre-trained CNN from disk

print("[INFO] loading model...")

model = Model(weights="imagenet")

model.summary()
# load the original image from disk (in OpenCV format) and then

# resize the image to its target dimensions

orig = cv2.imread(image_path)

resized = cv2.resize(orig, (224, 224))



# load the input image from disk (in Keras/TensorFlow format) and

# preprocess it

image = load_img(image_path, target_size=(224, 224))

image = img_to_array(image)

image = np.expand_dims(image, axis=0)

image = imagenet_utils.preprocess_input(image)



# use the network to make predictions on the input imag and find

# the class label index with the largest corresponding probability

preds = model.predict(image)

i = np.argmax(preds[0])



# decode the ImageNet predictions to obtain the human-readable label

decoded = imagenet_utils.decode_predictions(preds)

(imagenetID, label, prob) = decoded[0][0]

label = "{}: {:.2f}%".format(label, prob * 100)

print("[INFO] {}".format(label))



# initialize our gradient class activation map and build the heatmap

cam = GradCAM(model, i)

heatmap = cam.compute_heatmap(image)



# resize the resulting heatmap to the original input image dimensions

# and then overlay heatmap on top of the image

heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)



# draw the predicted label on the output image

cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)

cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,

	0.8, (255, 255, 255), 2)



# display the original image and resulting heatmap and output image

# to our screen

output = np.vstack([orig, heatmap, output])

output = cv2.resize(output, (2000,3000))

plt.imshow(output)

cv2.waitKey(0)