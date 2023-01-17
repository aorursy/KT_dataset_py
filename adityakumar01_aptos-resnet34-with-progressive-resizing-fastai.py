%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import fastai

from fastai.vision import *

import pandas as pd

import os

from shutil import copyfile



print(os.listdir("../input/aptos2019-blindness-detection"))
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



x_train = df_train['id_code']

y_train = df_train['diagnosis']
path = Path("../input/aptos2019-blindness-detection")
# Creating data into folder format

os.mkdir("../data")

os.mkdir("../data/train")

for i in range(5):

    os.mkdir("../data/train/"+str(i))

def make_img_folder(x, y):

    for img_name ,diagnosis in zip(x,y):

        if diagnosis == 0:

            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/0/{}.png'.format(img_name))

        if diagnosis == 1:

            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/1/{}.png'.format(img_name))

        if diagnosis == 2:

            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/2/{}.png'.format(img_name))

        if diagnosis == 3:

            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/3/{}.png'.format(img_name))

        if diagnosis == 4:

            copyfile('../input/aptos2019-blindness-detection/train_images/{}.png'.format(img_name), '../data/train/4/{}.png'.format(img_name))

            

make_img_folder(x_train, y_train)
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/pytorch-pretrained-image-models/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'
np.random.seed(42)

src = (ImageList.from_folder("../data/train")

                .split_by_rand_pct(0.2)

                .label_from_folder())

# Starting with image size 128

data = (src.transform(tfms=get_transforms(flip_vert=True, max_warp=0), size=128)

           .databunch(bs=64, path='.').normalize(imagenet_stats))
print(data.classes, data.c)
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir=".", callback_fns=ShowGraph)
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(5,3e-3)
learn.save("retino_128_1")

learn.load("retino_128_1")
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-5))
learn.save("retino128_2")
# Creating new data with image size increased to 256

data = (src.transform(tfms=get_transforms(flip_vert=True, max_warp=0), size=256)

           .databunch(bs=64, path='.').normalize(imagenet_stats))
learn.data = data
learn.fit_one_cycle(4, 1e-3)
learn.save("retino256_1")
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-6))
learn.save("retino256_2")
sample_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,'../input/aptos2019-blindness-detection/',folder='test_images',suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)