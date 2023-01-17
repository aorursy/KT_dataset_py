import numpy as np

import pandas as pd

import glob

import PIL

from PIL import Image

from warnings import filterwarnings

filterwarnings('ignore')
import re

numbers = re.compile(r'(\d+)')

def numericalSort(value):

    parts = numbers.split(value)

    parts[1::2] = map(int, parts[1::2])

    return parts
filelist_trainx = sorted(glob.glob('../input/*/trainx/*.bmp'), key=numericalSort)

X_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainx])



filelist_trainy = sorted(glob.glob('../input/*/trainy/*.bmp'), key=numericalSort)

Y_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainy])

size = 224,224

for i in range(0,200):

    im = Image.open(filelist_trainx[i])

    im_resized = im.resize(size, Image.ANTIALIAS)

    im_resized.save('X_img_'+str(i)+'.bmp', dpi = (224,244))
size = 224,224

for i in range(0,200):

    im = Image.open(filelist_trainy[i])

    im_resized = im.resize(size, Image.ANTIALIAS)

    im_resized.save('Y_img_'+str(i)+'.bmp', dpi = (224,224))
Image.open('X_img_0.bmp')