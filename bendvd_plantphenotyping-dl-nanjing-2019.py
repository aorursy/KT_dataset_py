# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from fastai.vision import *

import fastai

import pathlib

from fastai import *

from fastai.vision import *

from fastai.tabular import *

from torchvision.transforms import *



from itertools import product

import cv2

from tqdm import tqdm_notebook as tqdm



def dense_patchify(img, patch_size,min_overlap=0.1,scale=1 ):



    i_h, i_w, channel = np.shape(img)

    p_h, p_w = patch_size



    if i_h < p_h | i_w < p_w:

        print("image is too small for this patch size. Please add black pixels or resize images")



    if scale != 1:

        img = cv2.resize(img, (round(i_w*scale),round(i_h*scale)))







    n_h = 1

    n_w =1

    c_h, c_w = (i_h-p_h),(i_w-p_h)



    while c_h > p_h*(1-min_overlap):

        n_h += 1

        c_h = round((i_h-p_h)/n_h)



    while c_w > p_w*(1-min_overlap):

        n_w += 1

        c_w = round((i_w-p_w)/n_w)





    new_img = np.zeros(((c_h*n_h)+p_h+n_h,(c_w*n_w)+p_w+n_w,channel))

    new_img[:i_h,:i_w]=img

    

    possible_h = range(0,(c_h*(n_h+1)), c_h )

    possible_w = range(0,(c_w*(n_w+1)), c_w )



    origins_array = list(product(possible_h, possible_w))



    patches  = np.zeros((len(origins_array), p_h, p_w,3))

    

    for idx,origin in enumerate(origins_array):

        patches[idx,:,:,:] = cv2.resize(new_img[origin[0]:origin[0]+p_h,origin[1]:origin[1]+p_w,:], (p_h,p_w),  interpolation = cv2.INTER_AREA) 

            

    return patches, np.array(origins_array)

np.random.seed(84) # Here we fix the randomness to have replicable results





# Here are the path to dataset

data_dir = "/kaggle/input/v2-plant-seedlings-dataset/nonsegmentedv2/"

data_dir = pathlib.Path(data_dir)





data = ImageDataBunch.from_folder(data_dir, train='.', valid_pct=0.2,

                                  ds_tfms=(), size=224).normalize(imagenet_stats)





data.show_batch(rows=5, figsize=(15, 15))



# Link to doc : https://docs.fast.ai/vision.transform.html#get_transforms



tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_zoom=1.3, max_lighting=0.5, max_warp=0.1, p_affine=0.75, p_lighting=0.75)

data = ImageDataBunch.from_folder(data_dir, train='.', valid_pct=0.2,

                                  ds_tfms=tfms, size=224, bs=6).normalize(imagenet_stats)

data.show_batch(rows=5, figsize=(15, 15))
learn = cnn_learner(data,

                    models.resnet18,

                    loss_func =CrossEntropyFlat(),

                    opt_func=optim.Adam,

                    metrics=accuracy,

                    callback_fns=ShowGraph)



learn.model_dir = "/kaggle/models"
defaults.device = torch.device('cuda') # makes sure the gpu is used

learn.fit_one_cycle(4) # Here we fit the head for 4 epochs
learn.unfreeze() # must be done before calling lr_find

learn.lr_find()

learn.recorder.plot()



learn.fit_one_cycle(4, max_lr=slice(3e-6, 3e-5))

learn.save("plant-model")


preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_confusion_matrix(figsize=(10,10))
for i in interp.top_losses(10).indices:

    data.open(data.items[i]).show(title=data.items[i])



window_shape = (512, 512)

#B = extract_patches_2d(A, window_shape)







data_dir = "/kaggle/input/plant-segmentation/plant_segmentation/dataset/arabidopsis"

data_dir = pathlib.Path(data_dir)



path_img = data_dir/'images'



path_lbl = data_dir/'masks'



new_train_images = pathlib.Path() / "/kaggle/input/images"

new_train_masks = pathlib.Path() / "/kaggle/input/masks"



new_train_images.mkdir(exist_ok=True)

new_train_masks.mkdir(exist_ok=True)



for i,j  in tqdm(zip(path_img.glob("*"),path_lbl.glob("*"))):

    img = cv2.imread(str(i))

    patches,_ = dense_patchify(img,(512,512))

    

    for idx, p in enumerate(patches):

        cv2.imwrite(str( new_train_images / f"{str(idx)}_{i.name}"),p)



    img = cv2.imread(str(j))

    patches,_ = dense_patchify(img,(512,512))

    

    for idx, p in enumerate(patches):

        cv2.imwrite(str( new_train_masks / f"{str(idx)}_{j.name}"),p)

tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180.0, max_zoom=1.3, max_lighting=0.5, max_warp=0.1, p_affine=0.75, p_lighting=0.75) 



get_y_fn = lambda x: new_train_masks/f'{x.name}'

codes = ["Background","hypocotyl","non-hypocotyl"]



data = (SegmentationItemList.from_folder(new_train_images)

        .split_by_rand_pct()

        .label_from_func(get_y_fn, classes=codes)

        .transform(tfms,tfm_y=True)

        .databunch(bs=4, path=data_dir)

        .normalize(imagenet_stats))



data.show_batch(rows=2, figsize=(10, 10))



# https://towardsdatascience.com/introduction-to-image-augmentations-using-the-fastai-library-692dfaa2da42
wd=1e-2

learn = unet_learner(data, models.resnet18, metrics=dice, wd=wd)

learn.model_dir = "/kaggle/models"