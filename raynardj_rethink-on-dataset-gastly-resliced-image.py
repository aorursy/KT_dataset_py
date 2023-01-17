

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
!pip install -q forgebox
from forgebox.imports import *

from joblib import Parallel, delayed

from random import choice
INPUT = Path('/kaggle/input')

DATA = INPUT/"siim-isic-melanoma-classification"

TRAIN = DATA/"jpeg"/'train'

TEST  = DATA/"jpeg"/'test'
DATA.ls()
TRAIN.ls()[:5]
TEST.ls()[:5]


def open_img(path,parent = TRAIN):

    return Image.open(parent/path).convert('RGB')
open_img('ISIC_2679975.jpg').resize((300,300))
# resize/ rotate image in to the proper range, with w>h is a must

def proper_size(img):

    w,h = img.size

    if h>w:

        img = img.transpose(Image.ROTATE_90)

    check = False

    while check == False:

        if min(w,h)>1599:

            img = img.resize((w//2,h//2))

        if min(w,h)<256:

            img = img.resize((w*2,h*2))

            

        w,h = img.size

        check = True

    return img
# train_meta_df = pd.DataFrame(dict(fname = TRAIN.ls()))

# test_meta_df = pd.DataFrame(dict(fname = TEST.ls()))



# train_meta_df['img_size'] = train_meta_df.fname.apply(lambda fname: open_img(fname).size)



# test_meta_df['img_size'] = test_meta_df.fname.apply(lambda fname: open_img(fname,parent=TEST).size)



# train_meta_df.vc("img_size").head(20)



# test_meta_df.vc("img_size").head(20)
def find_center(img,size = 256):

    w,h = img.size

    left = w//2-size//2

    upper = h//2-size//2

    right = left+size

    lower = upper+size

    return img.crop((left, upper, right, lower))



def find_ratio(img,size = 256,ratio = .3):

    w,h = img.size

    h2 = int(h*ratio)

    upper = (h-h2)//2

    lower = upper+h2

    

    w2 = int(w*ratio)

    wpad = (w-w2)//2

    start = choice(list(range(max(1,w2-h2))))

    left = wpad+start

    right = left+h2



    return img.crop((left, upper, right, lower)).resize((size,size))



def combine_4in1(*imgs,size = 256):

    """

    combining 4 images of 'size' into image (2*size x 2*size)

    """

    dst = Image.new('RGB', (size*2,size*2))

    dst.paste(imgs[0], (0, 0))

    dst.paste(imgs[1], (0, size))

    dst.paste(imgs[2], (size, 0))

    dst.paste(imgs[3], (size, size))

    return dst



def different_scale_crop(img,size=512):

    """

    process for 4 shots and combine into 1

    """

    img = proper_size(img)

    return combine_4in1(*map(lambda i:find_ratio(img,size=size//2,ratio = 1-2*(i/10)),range(1,5)))
for i in range(100,150):

    img = different_scale_crop(open_img(TRAIN.ls()[i*2]),size = 512)

    display(img)
HOME = Path(".")



TRAIN_SAVE = HOME/"img/train"

TEST_SAVE = HOME/"img/test"



# !mkdir -p {TRAIN_SAVE}

# !mkdir -p {TEST_SAVE}


train_files = TRAIN.ls()

def process_file_train(fname):

    img = open_img(fname,parent = TRAIN)

    img = different_scale_crop(img,size = 512)

    newname = fname.split(".")[0]+".jpg"

    img.save(TRAIN_SAVE/f"{newname}")


test_files = TEST.ls()

def process_file_test(fname):

    img = open_img(fname,parent = TEST)

    img = different_scale_crop(img,size = 512)

    newname = fname.split(".")[0]+".jpg"

    img.save(TEST_SAVE/f"{newname}")
# Parallel(n_jobs=8)(delayed(process_file_train)(fname) for fname in train_files)



# Parallel(n_jobs=8)(delayed(process_file_test)(fname) for fname in test_files)



# !ls -l {TRAIN_SAVE}|wc -l



# !ls -l {TEST_SAVE}|wc -l
# !tar -czvf train_data.tar.gz {TRAIN_SAVE} > /dev/null



# !rm -rf {TRAIN_SAVE}



# !tar -czvf test_data.tar.gz {TEST_SAVE} > /dev/null



# !rm -rf {TEST_SAVE}