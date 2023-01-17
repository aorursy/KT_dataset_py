import os

from pathlib import Path

import numpy as np

import pandas as pd

import numpy as np

from PIL import Image

from fastai.vision.all import *



in_folder = Path("../input/digit-recognizer")
### Function that converts all columns starting with pixel into img and returns others column and pixels replaced with img

def get_images(df): 

    IMG_WIDTH = 28

    IMG_HEIGHT = 28

    

    df['img'] = df[df.columns[df.columns.str.startswith('pixel')]].apply(

        lambda x : PILImage(Image.fromarray(np.array(x.values).reshape((IMG_WIDTH, IMG_HEIGHT)).astype(np.uint8))),axis=1)

    

    return df[df.columns[[not x for x in df.columns.str.startswith('pixel')]]]
train_df = pd.read_csv(in_folder/"train.csv")

train_df = get_images(train_df)

#train_df.head()
def get_x(r): return r['img']

def get_y(r): return r['label']

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock), get_x = get_x, get_y = get_y)

dls = dblock.dataloaders(train_df)
learn = cnn_learner(dls, resnet34, metrics=error_rate).to_fp16()

learn.fine_tune(20)
test_df = pd.read_csv(in_folder/"test.csv")

test_df['ImageId'] = test_df.index+1

test_df = get_images(test_df)

#test_df.head()
#test_df['img1'] = test_df['img'].apply(lambda x : PILImage(x))

dl = learn.dls.test_dl(list(test_df['img']))

inp,preds,_,dec_preds = learn.get_preds(dl=dl, with_input=True, with_decoded=True)

test_df['Label'] = dec_preds
submission = test_df[['ImageId','Label']]

submission.to_csv("submission.csv", index = False)