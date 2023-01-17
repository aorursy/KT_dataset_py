# import required packages

import numpy as np

import pandas as pd

from pathlib import Path

from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.utils.mem import *

from progressbar import ProgressBar

import cv2

import os

import json
# create a folder for the mask images

if  not os.path.isdir('../labels'):

    os.makedirs('../labels')
path = Path("../input")

path_img = path/'train'

path_lbl = Path("../labels")



# only the 27 apparel items, plus 1 for background

# model image size 224x224

category_num = 27 + 1

size = 224



# get and show categories

with open(path/"label_descriptions.json") as f:

    label_descriptions = json.load(f)



label_names = [x['name'] for x in label_descriptions['categories']]

print(label_names)



# train dataframe

df = pd.read_csv(path/'train.csv')
# training jpg images are in the train folder

fnames = get_image_files(path_img)

print(fnames[0])
# need a function to turn the run encoded pixels from train.csv into an image mask

# there are multiple rows per image for different apparel items, this groups them into one mask

def make_mask_img(segment_df):

    seg_width = segment_df.at[0, "Width"]

    seg_height = segment_df.at[0, "Height"]

    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)

    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):

        pixel_list = list(map(int, encoded_pixels.split(" ")))

        for i in range(0, len(pixel_list), 2):

            start_index = pixel_list[i] - 1

            index_len = pixel_list[i+1] - 1

            if int(class_id.split("_")[0]) < category_num - 1:

                seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])

    seg_img = seg_img.reshape((seg_height, seg_width), order='F')

    return seg_img
# we can look at an image to see how the processing works

# the original image

img_file = fnames[500]

img = open_image(img_file)

img.show(figsize=(5,5))
# convert rows for this image into a numpy array mask 

img_name = os.path.basename(img_file)

img_df = df[df.ImageId == img_name].reset_index()

#img_df = img_df.iloc[0:1]

#img_df = img_df[img_df.ClassId.astype(int) < category_num - 1].reset_index()

img_mask = make_mask_img(img_df)

plt.imshow(img_mask)
# convert the numpy array into a three channel png that can be used in the standard SegmentationItemList

# then write into the labels folder as png and show the image

# all pixels have the category numbers, so it looks like a dark greyscale image

img_mask_3_chn = np.dstack((img_mask, img_mask, img_mask))

cv2.imwrite('../labels/' + os.path.splitext(img_name)[0] + '_P.png', img_mask_3_chn)

png = open_image('../labels/' + os.path.splitext(img_name)[0] + '_P.png')

png.show(figsize=(5,5))
# use fastai's open_mask for an easier-to-view image (and check it works...)

mask = open_mask('../labels/' + os.path.splitext(img_name)[0] + '_P.png')

mask.show(figsize=(5,5), alpha=1)

print(mask.data)
# run the same procedure for a sample of first 5000 images in dataset

images = df.ImageId.unique()[:5000]
pbar = ProgressBar()



for img in pbar(images):

    img_df = df[df.ImageId == img].reset_index()

    img_mask = make_mask_img(img_df)

    img_mask_3_chn = np.dstack((img_mask, img_mask, img_mask))

    cv2.imwrite('../labels/' + os.path.splitext(img)[0] + '_P.png', img_mask_3_chn)
# before creating the databunch we need a function to find the mask images 

# also set the batch size, categories and wd

get_y_fn = lambda x: path_lbl/f'{Path(x).stem}_P.png'

bs = 32

#classes = label_names

codes = list(range(category_num))

wd = 1e-2
# create the databunch

images_df = pd.DataFrame(images)



src = (SegmentationItemList.from_df(images_df, path_img)

       .split_by_rand_pct()

       .label_from_func(get_y_fn, classes=codes))



data = (src.transform(get_transforms(), size=size, tfm_y=True)

       .databunch(bs=bs)

       .normalize(imagenet_stats))
# look at a batch

data.show_batch(3, figsize=(10,10))
# I create an accuracy metric which excludes the background pixels

# not sure if this is correct

def acc_fashion(input, target):

    target = target.squeeze(1)

    mask = target != category_num - 1

    return (input.argmax(dim=1)==target).float().mean()
# learner, include where to save pre-trained weights (default is in non-write directory)

learn = unet_learner(data, models.resnet34, metrics=acc_fashion, wd=wd, model_dir="/kaggle/working/models")
# run learning rate finder

lr_find(learn)

learn.recorder.plot()
# set learning rate based on roughly the steepest part of the curve

lr=1e-4
# train for 10 cycles frozen

learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
# take a look at some results

learn.show_results()
# unfreeze earlier weights

learn.unfreeze()
# decrease the learning rate

lrs = slice(lr/400,lr/4)
# train for 10 more cycles unfrozen

learn.fit_one_cycle(10, lrs, pct_start=0.8)
# more results

learn.show_results()