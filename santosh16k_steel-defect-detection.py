%reload_ext autoreload

%autoreload 2

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai import *

import pandas as pd

import matplotlib.pyplot as plt
path = Path('/kaggle/input/severstal-steel-defect-detection/')

train_path = 'train_images'

train_csv = Path('train.csv')
df = pd.read_csv(path/train_csv)

df.head(5)
# !cat /kaggle/input/severstal-steel-defect-detection/train.csv

len(df)
ndefects = 0

n_nodefects = 0

for col in range(0, len(df), 4):

    image_name = [str(i).split("_")[0] for i in df.iloc[col:col+4, 0].values]

    if not (image_name[0] == image_name[1] == image_name[2] == image_name[3]):

        raise ValueError

    

    labels = df.iloc[col:col+4, 1]

    

    if labels.isna().all():

        n_nodefects += 1

    else:

        ndefects += 1
print(n_nodefects)

print(ndefects)
path.ls()
fnames = get_image_files(path/train_path)
img_1 = fnames[11]

img = open_image(img_1)

img.show()
df["img_id"] = df["ImageId_ClassId"].str.split(".").str[0]

df["class_id"] =  df["ImageId_ClassId"].str.split(".").str[1].str.split('_').str[1]
df.head(20)
!mkdir ../masks
import imageio

from tqdm import tqdm



for img_id in tqdm(set(df["img_id"])):

    tmp_df = df.loc[lambda d: d["img_id"] == img_id, ["EncodedPixels", "class_id"]]

#     print(tmp_df)



    if tmp_df["EncodedPixels"].isnull().all():

        img_path ='/kaggle/input/severstal-steel-defect-detection/train_images/{}.jpg'.format(img_id)

    

        img = open_image(img_path)

        

        pure = np.zeros((img.px.shape[1], img.px.shape[2]), dtype=np.uint8)

        imageio.imwrite(f"../masks/{img_id}.jpg", pure)

        continue



        

#     mask_rle = df.loc[lambda d: d["img_id"] == img_id, "EncodedPixels"].values

    mask_rle1 = df.loc[lambda d: d["img_id"] == img_id, "EncodedPixels"].notnull()

    tmp_df = df.loc[lambda d: d["img_id"] == img_id, "EncodedPixels"]

    mask_rle = tmp_df[mask_rle1 == True].values[0]

    



    img_path ='/kaggle/input/severstal-steel-defect-detection/train_images/{}.jpg'.format(img_id)

    

    img = open_image(img_path)

    mask_shape = (img.px.shape[1], img.px.shape[2])

    mask = open_mask_rle(mask_rle, shape=mask_shape)

    mask = ImageSegment(mask.data.transpose(2, 1))

    

    mask.save(f"../masks/{img_id}.jpg")

    

#     img.show(y=mask, figsize=(20, 10), title=f"{img_id} with mask, label 1")

    

masks_path = Path('/kaggle/masks')

train_path = Path('/kaggle/input/severstal-steel-defect-detection/train_images')
path.ls()

fnames = get_image_files('/kaggle/input/severstal-steel-defect-detection/train_images')

lbl_names = get_image_files(masks_path)
lbl_names[:3]

fnames[:3]
# !ls /kaggle/input/severstal-steel-defect-detection/train_image

img_f = Path('2e5dab72b.jpg')



img = open_image(train_path/img_f)

img.show(figsize=(15,15))
# def get_y_fn(img): return "{}".format(img)

def get_y_fn(x): return masks_path/x.name

# codes = array(['0', '1', '2', '3', '4'])

codes = [0, 1, 2, 3, 4]
print(img_f)

print(get_y_fn(img_f))

mask = open_mask(get_y_fn(img_f))

mask.show(figsize=(15,15), alpha=1)
src = (SegmentationItemList.from_folder(train_path)

       .split_by_rand_pct()

       .label_from_func(get_y_fn, classes=codes))
mask.shape[1]
bs = 16

size = mask.shape[1]

data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
data.show_batch(2, figsize=(50,35))


name2id = {v:k for k,v in enumerate(codes)}

void_code = name2id[0]



# def acc_camvid(input, target):

#     target = target.squeeze(1)

#     mask = target != void_code

#     return (input.argmax(dim=1)[mask]==target[mask]).float().mean()



def dice(pred, targs):

    pred = (pred>0).float()

    return 2. * (pred*targs).sum() / (pred+targs).sum()
name2id
metrics=dice

wd=1e-2
models.resnet34
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, bottle=True, model_dir="/kaggle/working")
!mkdir ../model

# torch.cuda.empty_cache()
learn.model
learn.model_dir = "/kaggle/model"

lr_find(learn)
# learn.recorder.plot()
lr = 0.001
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)