import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

# ../input/img_align_celeba/img_align_celeba

# 000001.jpg  033768.jpg	067535.jpg  101302.jpg	135069.jpg  168836.jpg ...
import matplotlib.pyplot as plt

from PIL import Image, ImageColor

image = Image.open('../input/img_align_celeba/img_align_celeba/000001.jpg')
# im = Image.open('../input/img_align_celeba/img_align_celeba/000001.jpg')

# im.show('../input/img_align_celeba/img_align_celeba/000001.jpg')