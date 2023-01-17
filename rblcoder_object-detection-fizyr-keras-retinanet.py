import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

print(os.listdir("../input/test/")[:1])

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
!git clone https://github.com/fizyr/keras-retinanet.git
%cd keras-retinanet/



!pip install .
!python setup.py build_ext --inplace


%load_ext autoreload

%autoreload 2



# import keras

import keras



# import keras_retinanet

from keras_retinanet import models

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet.utils.visualization import draw_box, draw_caption

from keras_retinanet.utils.colors import label_color

from keras_retinanet.utils.gpu import setup_gpu



# import miscellaneous modules

import matplotlib.pyplot as plt

import cv2

import os

import numpy as np

import time



# set tf backend to allow memory to grow, instead of claiming everything

import tensorflow as tf



def get_session():

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    return tf.Session(config=config)



# use this to change which GPU to use

gpu = 0



# set the modified tf session as backend in keras

setup_gpu(gpu)
!wget 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'

!ls
model_path = 'resnet50_coco_best_v2.1.0.h5'

model = models.load_model(model_path, backbone_name='resnet50')

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
image = read_image_bgr('../../input/test/da4245f6018ea0a2.jpg')

draw = image.copy()

draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



# preprocess image for network

image = preprocess_image(image)

image, scale = resize_image(image)



# process image

start = time.time()

boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

print("processing time: ", time.time() - start)



# correct for image scale

boxes /= scale



# visualize detections

for box, score, label in zip(boxes[0], scores[0], labels[0]):

    # scores are sorted so we can break

    if score < 0.5:

        break

        

    color = label_color(label)

    

    b = box.astype(int)

    draw_box(draw, b, color=color)

    

    caption = "{} {:.3f}".format(labels_to_names[label], score)

    draw_caption(draw, b, caption)

    

plt.figure(figsize=(15, 15))

plt.axis('off')

plt.imshow(draw)

plt.show()
df_test = pd.DataFrame(os.listdir("../../input/test/"), columns=['file'])

df_test.info()
df_test.iloc[:10, 0].values
df_test.iloc[:, 0] = "../../input/test/" + df_test.iloc[:, 0]
#https://www.kaggle.com/vikramtiwari/baseline-predictions-using-inception-resnet-v2

#image_paths = get_images(10)

images = df_test.iloc[:80, 0].values

predictions = []

ImageID = []

PredictionString = []

for image_path in images:

    print(image_path)

    image = read_image_bgr(image_path)

    draw = image.copy()

    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



    # preprocess image for network

    image = preprocess_image(image)

    image, scale = resize_image(image)



    # process image

    start = time.time()

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    print("processing time: ", time.time() - start)

    # print(labels)

    # correct for image scale

    boxes /= scale

    # visualize detections

    for box, score, label in zip(boxes[0], scores[0], labels[0]):

        # scores are sorted so we can break

        if score < 0.5:

            break



        color = label_color(label)



        b = box.astype(int)

        draw_box(draw, b, color=color)



        caption = "{} {:.3f}".format(labels_to_names[label], score)

        draw_caption(draw, b, caption)



    plt.figure(figsize=(15, 15))

    plt.axis('off')

    plt.imshow(draw)

    plt.show()



        
%cd ..
!rm -rf keras-retinanet