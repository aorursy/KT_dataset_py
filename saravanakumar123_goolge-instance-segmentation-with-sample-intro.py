%matplotlib inline
from matplotlib import pyplot as plt

from gluoncv import model_zoo, data, utils
net = model_zoo.get_model('mask_rcnn_fpn_resnet101_v1d_coco', pretrained=True)
from PIL import Image

import pandas as pd

import os

import numpy as np
image_file = os.listdir("../input/test/")
pth = '../input/test/'+image_file[84822]

Image.open(pth)
x, orig_img = data.transforms.presets.rcnn.load_test(pth)



ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]



width, height = orig_img.shape[1], orig_img.shape[0]

masks = utils.viz.expand_mask(masks, bboxes, (width, height), scores)

orig_img = utils.viz.plot_mask(orig_img, masks)



# identical to Faster RCNN object detection

fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(1, 1, 1)

ax = utils.viz.plot_bbox(orig_img, bboxes, scores, ids,

                         class_names=net.classes, ax=ax)

plt.show()