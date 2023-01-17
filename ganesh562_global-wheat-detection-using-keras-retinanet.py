# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import  scipy.misc as smi

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import glob

from PIL import Image

import cv2

import numpy as np

from sklearn.model_selection import train_test_split

import sys

import keras

import keras.preprocessing.image

import tensorflow as tf
os.chdir("/kaggle/input/kerasresnet/keras-resnet")

!pip install .

os.chdir("/kaggle/input/pythonutils/python-utils/")

!pip install .

os.chdir("/kaggle/input/progressbar2/python-progressbar/")

!pip install .

os.chdir("/kaggle/input/keras-retinanet/keras-retinanet")

!pip install .
!mv "/kaggle/input/keras-retinanet" "/kaggle/working/"

os.chdir("/kaggle/working/keras-retinanet/keras-retinanet")

!python setup.py build_ext --inplace
# !git clone https://github.com/SriGanesh130/keras-retinanet.git

# os.chdir("keras-retinanet") 

# !python setup.py build_ext --inplace
# !mv "/kaggle/working/keras-retinanet" "/kaggle/input/"
# retina_net_dir = "/kaggle/working/keras-retinanet"

# !cd {retina_net_dir} && pip install .
from keras_retinanet.models import load_model
os.chdir("/kaggle/working")

model = load_model("../input/weights101/resnet_infer101_csv_37.h5", backbone_name="resnet101")
input_dir = "/kaggle/input/"

output_dir = "/kaggle/working/"
b_box = pd.read_csv(os.path.join(input_dir, "global-wheat-detection/train.csv"))

image_path = input_dir + "global-wheat-detection/train/"
b_box["source"].value_counts(normalize = True).plot(kind = "bar")
b_box["image_id"].value_counts()
image_id = "b6ab77fd7"

image = cv2.imread(os.path.join(image_path, image_id + ".jpg")) #read image

print(image.shape)

img_same_kind = b_box[b_box["image_id"]==image_id] #read all test labels related to the image

lst = img_same_kind["bbox"].values[:] # get all the bounding boxes from the labels

contours = [img_index.strip('][').split(', ') for img_index in lst ] #casting bounding boxes from str to lst

print(len(contours))

plt.figure()

plt.imshow(image) 

plt.show() 


#plotting bounding boxes on image

def plot_boundbox(res, image):

    x = int(float(res[0]))

    y = int(float(res[1]))

    w = int(float(res[2]))

    h = int(float(res[3]))

    bound_boxes = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    return bound_boxes



image_orig = image

print(len(contours))

for i, contour in enumerate(contours):

    image_new = plot_boundbox(contour, image_orig)

    image_orig = image_new

plt.figure()

plt.imshow(image_new) 

plt.show() 
# Creating Dataset for RetinaNet

def create_dataset_csv(data, image_path):

    annotated_data = []

    for ind in range(data.shape[0]):

        image_id = image_path + data["image_id"][ind] + '.jpg'

        b_box = data["bbox"][ind].strip('][').split(', ')

        class_name = "wheat"

        annotated_data.append([image_id, b_box[0], b_box[1], str(float(b_box[0]) + float(b_box[2])), 

                               str(float(b_box[1]) + float(b_box[3])), class_name])

    df = pd.DataFrame(annotated_data)

    return df
# train_csv = create_dataset_csv(b_box, "./images/data/train/")

# class_map = pd.DataFrame([["wheat", 0]])
# train_csv.to_csv(output_dir + "train_data.csv")

# class_map.to_csv(output_dir + "class_map.csv")
from keras_retinanet import models

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet.utils.visualization import draw_box, draw_caption

from keras_retinanet.utils.colors import label_color

from keras_retinanet.utils.gpu import setup_gpu
labels_to_names = {0:'wheat'}
def format_prediction_string(boxes, scores):

    pred_strings = []

    for j in zip(scores, boxes):

        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], int(j[1][0]), int(j[1][1]), 

                                                             int(j[1][2]), int(j[1][3])))



    return " ".join(pred_strings)
# image = read_image_bgr('./test/' + test_image)

test_dir = "../input/global-wheat-detection/test/"

test_images = os.listdir(test_dir)

results = []

for test_image in test_images:

    image = read_image_bgr(os.path.join(test_dir, test_image))

    draw = image.copy()

    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)

    image, scale = resize_image(image)

#     import pdb

#     pdb.set_trace()

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    boxes /= scale

    boxes = np.squeeze(boxes)

    scores = np.squeeze(scores)

    boxes_f = boxes[scores > 0.4]

    scores_f = scores[scores >0.4]

    boxes_f[:, 2] = boxes_f[:, 2] - boxes_f[:, 0]

    boxes_f[:, 3] = boxes_f[:, 3] - boxes_f[:, 1]

    result = {"image_id" : test_image.split('.')[0], 

            "PredictionString": format_prediction_string(boxes_f, scores_f)}

    results.append(result)
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df.to_csv('submission.csv', index=False)

print(test_df)
d = pd.read_csv("./submission.csv")
d.head(5)