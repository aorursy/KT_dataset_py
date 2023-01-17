# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

from six import BytesIO

import numpy as np

import xml.etree.ElementTree as et

import ast

import tqdm

from itertools import chain

from xml.dom import minidom

from PIL import Image

from PIL import ImageColor

from PIL import ImageDraw

from PIL import ImageFont

from PIL import ImageOps

import cv2

import glob

import time
path1='/kaggle/input/open-images-object-detection-rvc-2020/test/'

sample = pd.read_csv("/kaggle/input/open-images-object-detection-rvc-2020/sample_submission.csv")

sample.head()

ids = []

for i in range(len(sample)):

    ids.append(sample['ImageId'][i])
#ids[0:5
img_data=[]

for i in range(len(sample)):

    img_data.append(glob.glob('/kaggle/input/open-images-object-detection-rvc-2020/test/{0}.jpg'.format(ids[i])))
img_data[0:5]
def get_prediction_string(result):

    with tf.device('/device:GPU:0'):

        df = pd.DataFrame(columns=['Ymin','Xmin','Ymax', 'Xmax','Score','Label','Class_label','Class_name'])

        min_score=0.01

        for i in range(result['detection_boxes'].shape[0]):

           if (result["detection_scores"][i]) >= min_score:

              df.loc[i]= tuple(result['detection_boxes'][i])+(result["detection_scores"][i],)+(result["detection_class_labels"][i],)+(result["detection_class_names"][i],)+(result["detection_class_entities"][i],)

        return df
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

with tf.device('/device:GPU:0'):

    with tf.Graph().as_default():

        detector = hub.Module(module_handle)

        image_string_placeholder = tf.placeholder(tf.string)

        decoded_image = tf.image.decode_jpeg(image_string_placeholder)

        decoded_image_float = tf.image.convert_image_dtype(

            image=decoded_image, dtype=tf.float32)

        module_input = tf.expand_dims(decoded_image_float, 0)

        result = detector(module_input, as_dict=True)

        init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]



        session = tf.Session()

        session.run(init_ops)
def nms(dets, thresh):

    x1 = dets[:, 0]

    y1 = dets[:, 1]

    x2 = dets[:, 2]

    y2 = dets[:, 3]

    scores = dets[:, 4]



    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]



    keep = []

    while order.size > 0:

        i = order[0]

        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])

        yy1 = np.maximum(y1[i], y1[order[1:]])

        xx2 = np.minimum(x2[i], x2[order[1:]])

        yy2 = np.minimum(y2[i], y2[order[1:]])



        w = np.maximum(0.0, xx2 - xx1 + 1)

        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)



        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]



    return keep
image_paths = img_data[0:10]

images = []

for f in image_paths:

    images.append(np.asarray(Image.open(f[0])))
!mkdir detect
image_id = sample['ImageId']

def format_prediction_string(image_id, result):

    prediction_strings = []

    

    for i in range(len(result['Score'])):

        class_name = result['Class_label'][i].decode("utf-8")

        YMin,XMin,YMax,XMax = result['Ymin'][i],result['Xmin'][i],result['Ymax'][i],result['Xmax'][i]

        score = result['Score'][i]

        

        prediction_strings.append(

            f"{class_name} {score} {XMin} {YMin} {XMax} {YMax}"

        )

        

    prediction_string = " ".join(prediction_strings)



    return {

        "PredictionString": prediction_string

    }
k =-1

predictions = []

with tf.device('/device:GPU:0'):

    for image_path in image_paths:

        k=k+1

        img_path = img_data[k]

        img = cv2.imread(img_path[0])

        with tf.io.gfile.GFile(image_path[0], "rb") as binfile:

            image_string = binfile.read()



        inference_start_time = time.time()

        result_out, image_out = session.run(

            [result, decoded_image],

            feed_dict={image_string_placeholder: image_string})

        df1=get_prediction_string(result_out)

        z1=nms(df1.values,0.68)

        z=df1.iloc[z1]

        z=z.reset_index()

        predictions.append(format_prediction_string(image_id, z))

        data1=z

        COLORS = np.random.uniform(0, 255, size=(len(z['Class_name']), 3))

        for m in range(len(data1)):

            if data1['Score'][m] >=0.01:

                img_class=data1.iloc[m].Class_name

                img_xmax, img_ymax =images[k].shape[1],images[k].shape[0]

                bbox_x_max, bbox_x_min = data1.Xmax[m] * img_xmax, data1.Xmin[m] * img_xmax

                bbox_y_max ,bbox_y_min = data1.Ymax[m] * img_ymax, data1.Ymin[m] * img_ymax

                xmin = int(bbox_x_min)

                ymin = int(bbox_y_min)

                xmax = int(bbox_x_max)

                ymax = int(bbox_y_max)

                width = xmax - xmin

                height = ymax - ymin

                label = str(data1['Class_name'][m])

                color = COLORS[m]

                cv2.rectangle(img, (xmin, ymax), (xmax, ymin), color, 2)

                path1 = '/kaggle/working/detect/'+str(k)+'.jpg'

                cv2.imwrite(path1, img)

                cv2.putText(img, label, (xmax,ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9,color, 2)
def load_images(folder):

    images = []

    for filename in os.listdir(folder):

        img = Image.open(os.path.join(folder, filename))

        if img is not None:

            images.append(img)

    return images
z = load_images("/kaggle/working/detect")

z[0]
z[1]
z[3]