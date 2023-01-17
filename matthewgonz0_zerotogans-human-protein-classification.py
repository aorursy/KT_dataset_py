import pandas as pd

from fastai.vision import *

import torch

import os

import warnings

warnings.simplefilter("ignore", UserWarning) #remove warnings when changing the size of the image
torch.cuda.is_available()
torch.backends.cudnn.enabled
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'



TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images

TEST_DIR = DATA_DIR + '/test'                             # Contains test images



TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images

TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'   # Contains dummy labels for test image
MODEL_DIR = os.makedirs('../models') #creates folder for fastai model saves to be stored
!head "{TRAIN_CSV}"
train_df = pd.read_csv(TRAIN_CSV)

train_df.head()
torch.device(0)
# tfms = get_transforms(do_flip=True, xtra_tfms = (zoom_crop(scale=(0.75,2), do_rand=True), rand_resize_crop(size=512)))

# tfms = get_transforms() #use default fastai transforms



data = ImageDataBunch.from_df(path=DATA_DIR, folder='train', label_delim=' ', 

                              df=train_df, suffix='.png', # ds_tfms=tfms, 

                              test='test',device=torch.device(0) #size=64

                             )

# data.normalize(imagenet_stats)
data.show_batch()
torch.cuda.get_device_name(0)
learn = cnn_learner(data, models.resnet34, metrics=[MultiLabelFbeta(average='macro')], path='../models')

learn.model.cuda() # run cnn learner on gpu



# learn.clip_grad()

learn.lr_find()

learn.recorder.plot(suggestion=True)

min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(10,min_grad_lr)

learn.save('res34_standard_norm')
preds, target = learn.get_preds(ds_type=DatasetType.Test)
def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)
submission_df = pd.read_csv(TEST_CSV)

submission_df.Label = [decode_target(i, threshold=0.2) for i in preds]

submission_df.head()

learn.show_results()
submission_df.to_csv('submission.csv', index=False)
!head "./submission.csv"