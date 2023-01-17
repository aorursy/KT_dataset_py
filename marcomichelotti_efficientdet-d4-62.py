# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
import json
import numpy as np
import os
import time
import glob

# forked a github repository to add fine tuning to efficient det
!cp -r ../input/repository-efficient/EfficientDet-master/* ./
!python setup.py build_ext --inplace
from model import efficientdet

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessio
def get_frozen_graph(graph_file):
    with tf.io.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def preprocess_image(image, image_size):
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale

def postprocess_boxes(boxes, scale, height, width):
    c_boxes = boxes.copy()
    c_boxes /= scale
    c_boxes[:, 0] = np.clip(c_boxes[:, 0], 0, width - 1)
    c_boxes[:, 1] = np.clip(c_boxes[:, 1], 0, height - 1)
    c_boxes[:, 2] = np.clip(c_boxes[:, 2], 0, width - 1)
    c_boxes[:, 3] = np.clip(c_boxes[:, 3], 0, height - 1)
    return c_boxes

# import keras
import numpy as np
from tensorflow import keras

from utils.compute_overlap import compute_overlap
from utils.anchors import anchors_for_shape


phi = 4
model_path = '/kaggle/input/efficientdetfrozen/efficientdet_0-80.pb'
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]
classes = [
    'wheat'
]
num_classes = len(classes)

score_threshold = 0.70

_, model = efficientdet(phi=phi,
                            weighted_bifpn=True,
                            num_classes=num_classes,
                            score_threshold=score_threshold,
                            finetuned_num_classes=1)

#model.load_weights(model_path, by_name=True)
#model.load_weights('/kaggle/input/efficientdetwheat-may/EfficientDet-23-0.8077.h5', by_name=True)
model.load_weights('/kaggle/input/efficientdet-04-084/EfficientDet-04-0.8407.h5', by_name=True)


result_data = []
for image_path in glob.glob('/kaggle/input/global-wheat-detection/test/*.jpg'):
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    src_image = image.copy()
    # BGR -> RGB
    image = image[:, :, ::-1]
    h, w = image.shape[:2]

    image, scale = preprocess_image(image, image_size=image_size)
    # run network
    start = time.time()
    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    print(time.time() - start)
    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

    # select indices which have a score above the threshold
    indices = np.where(scores[:] > score_threshold)[0]

    # select those detections
    boxes = boxes[indices]
    labels = labels[indices]
    
    row = [image_name.replace('.jpg','')]
    r_boxes = ""
    for s,b in zip(scores, boxes):
        if r_boxes != "":
            r_boxes += " "
        r_boxes += f"{round(float(s),2)} {int(b[0])} {int(b[1])} {int(b[2]-b[0])} {int(b[3]-b[1])}"
    
    row.append(r_boxes)
    
    print(row)
    result_data.append(row)
test_df = pd.DataFrame(result_data, columns=['image_id','PredictionString'])
test_df.to_csv("submission.csv", index=False)