!git clone https://github.com/NavaHub/headnet.git

import shutil

shutil.rmtree('headnet/.git')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import sys



print("Available data: ", os.listdir("../input/"))

print("Annotations of heads: ", os.listdir("../input/headsetannotationsint/"))

# Any results you write to the current directory are saved as output.
os.listdir("../input/headsset/hollywoodheads/HollywoodHeads/JPEGImages/")[:10]
sys.path.append('/input/headnet')

from headnet import models

from headnet.utils.transform import random_transform_generator

from headnet.preprocessing import CSVGenerator

from headnet import losses

from headnet.models.retinanet import retinanet_bbox

import keras



def model_with_weights(model, weights, skip_mismatch):

    if weights is not None:

        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)

    return model
from xml.dom.minidom import parse

import csv

import os



def writeToCSV(filename, xmin, ymin, xmax, ymax):

        csvfile = open('../input/data.csv', 'a+', newline='')

        thewriter = csv.writer(csvfile)

        thewriter.writerow(["../input/headsset/hollywoodheads/HollywoodHeads/JPEGImages/"+ filename, xmin, ymin, xmax, ymax, 'head'])

        

                

                

for root, dirs, files in os.walk('../input/headsset/hollywoodheads/HollywoodHeads/Annotations'):

    for file in files:

        if (file.endswith('.xml')):

            xmldoc = parse(os.path.join(root, file))

            filename = xmldoc.getElementsByTagName("filename")[0].firstChild.data

            i = 0

            while( i < xmldoc.getElementsByTagName("xmin").length):

                xmin = str(xmldoc.getElementsByTagName("xmin")[i].firstChild.data)

                ymin = str(xmldoc.getElementsByTagName("ymin")[i].firstChild.data)

                xmax = str(xmldoc.getElementsByTagName("xmax")[i].firstChild.data)

                ymax = str(xmldoc.getElementsByTagName("ymax")[i].firstChild.data)

                writeToCSV(filename, xmin, ymin, xmax, ymax)

                i = i + 1
with open('classes.csv', 'w') as __file:

        __file.write('head,0')



from shutil import copyfile

copyfile('../input/headsetannotationsint/Annotations.csv', "annotations.csv")
backbone = models.backbone('resnet50')

transform_generator = random_transform_generator(flip_x_chance=0.5)

train_generator = CSVGenerator(

    'annotations.csv',

    'classes.csv',

    transform_generator=transform_generator

)



weights = backbone.download_imagenet()

model = model_with_weights(backbone.retinanet(1, num_anchors=None, modifier=None), weights=weights, skip_mismatch=True)



model.compile(

    loss={

        'regression'    : losses.smooth_l1(),

        'classification': losses.focal()

    },

    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)

)



model.fit_generator(

    generator=train_generator,

    steps_per_epoch=200,

    workers=2,

    use_multiprocessing=True,

    max_queue_size=100,

    epochs=10,

    verbose=1

)
from matplotlib import pyplot as plt

from headnet.utils.image import read_image_bgr, preprocess_image, resize_image

from headnet.utils.visualization import draw_box, draw_caption

from headnet.utils.colors import label_color

import cv2



image = cv2.imread('../input/headsset/hollywoodheads/HollywoodHeads/JPEGImages/mov_001_117399.jpeg')



draw = image.copy()

draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



image = preprocess_image(image)

image, scale = resize_image(image)

prediction_model = retinanet_bbox(model=model, anchor_params=None)

boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))

boxes /= scale



for box, score, label in zip(boxes[0], scores[0], labels[0]):

    # scores are sorted so we can break

    if score < 0.5:

        break

        

    color = label_color(label)

    

    b = box.astype(int)

    draw_box(draw, b, color=color)

    

    caption = "{} {:.3f}".format('head', score)

    draw_caption(draw, b, caption)

    

plt.figure(figsize=(40, 40))

plt.axis('off')

plt.imshow(draw)

plt.show()