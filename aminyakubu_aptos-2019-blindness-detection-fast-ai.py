# loading the necessary libraries

import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai.metrics import error_rate

import os

print(os.listdir("../input"))
# Training

bs = 64 //2

path_anno = '../input/aptos2019-blindness-detection/train.csv'

path_img = '../input/aptos2019-blindness-detection/train_images'



# Test dataset

tpath_anno = '../input/aptos2019-blindness-detection/test.csv'

tpath_img = '../input/aptos2019-blindness-detection/test_images'
# creating directories and copying the models to those directories

!mkdir -p /tmp/.cache/torch/checkpoints

!cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth

!cp ../input/resnet152/resnet152.pth /tmp/.cache/torch/checkpoints/resnet152-b121ed2d.pth
# File names

fnames = get_image_files(path_img)

fnames[:5]
# Training images and their labels

df = pd.read_csv(path_anno)

df.head()
# Test images 

tdf = pd.read_csv(tpath_anno)

tdf.head()
test = ImageList.from_df(df = tdf, path = tpath_img, suffix = '.png')

len(test)
# Since these are microscopic images, I'll turn on random flipping of the images (since there's no really 

# up or down for the images) so the model generalizes well



tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1, max_warp=0.)
np.random.seed(42)

src = (ImageList.from_df(path = path_img, df = df, suffix = '.png')

        .split_by_rand_pct(0.2)

        .label_from_df(label_delim=None)

      .add_test(test))
data = (src.transform(tfms, size=126)

        .databunch().normalize(imagenet_stats))
data.classes
data.show_batch(rows=3, figsize=(7,6))
kappa = KappaScore()

kappa.weights = "quadratic"



learn = cnn_learner(data, models.resnet50, metrics=[error_rate, kappa], model_dir = Path('../kaggle/working'),

                   path = Path("."))



# Let's fit a couple of cycles

learn.fit_one_cycle(2)



# Now let's find a more accurate learning rate

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
# Let's fit some more with a more accurate learning rate

learn.fit_one_cycle(8, max_lr=slice(0.01))
# let's save the model in case, we want to return to this point

learn.save('train-1-rn50', return_path = True)
# let's see if we can get even a better learning rate from this point on

learn.load('train-1-rn50')



# Let's freeze up to last layer group. That is to sets every layer group except the last to untrainable 

learn.freeze()



# I'll try a smaller learning rate

learn.fit_one_cycle(8, max_lr=slice(1e-03))
data2 = (src.transform(tfms, size= 256)

        .databunch().normalize(imagenet_stats))
# Replacing data 

learn.data = data2



# train some more by building on the previous model

learn.fit_one_cycle(10)



# Learning rate

learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
# choosing a learning rate based on the data 

learn.fit_one_cycle(10, max_lr = slice(1e-4))
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

pred_prob, pred_class = preds.max(1)

len(pred_class)



my_submission = pd.DataFrame({'id_code': tdf['id_code'] , 'diagnosis': pred_class})

my_submission.to_csv('submission.csv', index=False)