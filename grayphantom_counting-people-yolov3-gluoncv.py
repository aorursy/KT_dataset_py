import mxnet as mx

from mxnet import image

from mxnet.gluon.data.vision import transforms

import gluoncv as gcv

import hashlib

from pylab import rcParams

from matplotlib import pyplot as plt

from gluoncv import model_zoo, data, utils

import numpy as np

import os

from pathlib import Path



rcParams['figure.figsize'] = 5, 10
M6_DATA = Path('../input/people-images')

M6_IMAGES = Path(M6_DATA, 'images')
network = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
def load_image(filepath):

    

    im=image.imread(filepath)

    return im

    
test_filepath = Path(M6_IMAGES, '32742378405_3ecc8cc958_b.jpg')

test_output = load_image(test_filepath)
plt.imshow(test_output.asnumpy())

fig = plt.gcf()

fig.set_size_inches(14, 14)

plt.show()
def transform_image(array):

    norm_image,image=data.transforms.presets.yolo.transform_test(array)

    return norm_image,image
norm_image, unnorm_image = transform_image(test_output)
plt.imshow(unnorm_image)

fig = plt.gcf()

fig.set_size_inches(14, 14)

plt.show()
def detect(network, data):

    pred=network(data)

    class_ids,scores,bounding_boxes=pred

    return class_ids, scores, bounding_boxes
class_ids, scores, bounding_boxes = detect(network, norm_image)
ax = utils.viz.plot_bbox(unnorm_image, bounding_boxes[0], scores[0], class_ids[0], class_names=network.classes)

fig = plt.gcf()

fig.set_size_inches(14, 14)

plt.show()
def count_object(network, class_ids, scores, bounding_boxes, object_label, threshold=0.75):

    idx=0

    for i in range(len(network.classes)):

        if network.classes[i]==object_label:

            idx=i

    scores=scores[0]

    class_ids=class_ids[0]

    num_people=0

    for i in range(len(scores)):

        proba=scores[i].astype('float32').asscalar()

        if proba>threshold and class_ids[i].asscalar()==idx:

            num_people+=1

    return num_people
for object_label in ["person", "sports ball"]:

    count = count_object(network, class_ids, scores, bounding_boxes, object_label)

    print("{} objects of class '{}' detected".format(count, object_label))
thresholds = [0, 0.5, 0.75, 0.9, 0.99, 0.999]

for threshold in thresholds:

    num_people = count_object(network, class_ids, scores, bounding_boxes, "person", threshold=threshold)

    print("{} people detected using a threshold of {}.".format(num_people, threshold))
class PersonCounter():

    def __init__(self, threshold):

        self._network = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)

        self._threshold = threshold



    def set_threshold(self, threshold):

        self._threshold = threshold

        

    def count(self, filepath, visualize=False):

        image=load_image(filepath)

        norm_image,unnorm_image=transform_image(image)

        network=self._network

        class_ids, scores, bounding_boxes = detect(network, norm_image)

        if visualize:

            self._visualize(unnorm_image, class_ids, scores, bounding_boxes)

        threshold=self._threshold

        object_label='person'

        num_people=count_object(network, class_ids, scores, bounding_boxes, object_label, threshold)

        if num_people == 1:

            print('{} person detected in {}.'.format(num_people, filepath.name)) 

        else:

            print('{} people detected in {}.'.format(num_people, filepath.name))

        return num_people

    

    def _visualize(self, unnorm_image, class_ids, scores, bounding_boxes):

        ax = utils.viz.plot_bbox(unnorm_image,

                                 bounding_boxes[0],

                                 scores[0],

                                 class_ids[0],

                                 class_names=self._network.classes)

        fig = plt.gcf()

        fig.set_size_inches(8,8)

        plt.show()
counter = PersonCounter(threshold=0.9)

counter.count(Path(M6_IMAGES, '31928213423_090ec29bcf_b.jpg')) 

counter.count(Path(M6_IMAGES, '32701657536_8a0d9e157f_b.jpg')) 

counter.count(Path(M6_IMAGES, '25751294956_fa3ee87fb8_b.jpg'))

counter.set_threshold(0.5)

counter.count(Path(M6_IMAGES, '31928213423_090ec29bcf_b.jpg'), visualize=True)

counter.count(Path(M6_IMAGES, '32701657536_8a0d9e157f_b.jpg'), visualize=True)

counter.count(Path(M6_IMAGES, '25751294956_fa3ee87fb8_b.jpg'), visualize=True)
counter.count(Path(M6_IMAGES, '18611133536_534285f26d_b.jpg'), visualize=True)
counter.count(Path(M6_IMAGES, '3354172257_a48ba3d1d8_b.jpg'), visualize=True)
counter.count(Path(M6_IMAGES, '2455110397_757dfec324_z.jpg'), visualize=True)
total_count = 0

for filepath in M6_IMAGES.glob('**/*.jpg'):

    total_count += counter.count(filepath)

print("### Summary: {} people detected.".format(total_count))