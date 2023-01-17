# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import IPython

import IPython.display

import PIL

import pickle



import torch

import torch.nn as nn

import torch.nn.functional as F



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
! ls ../input/fat2019_prep_mels1
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, 3, 1, 1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

        )



        self._init_weights()

        

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:

                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.zeros_(m.bias)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = F.avg_pool2d(x, 2)

        return x

    

class Classifier(nn.Module):

    def __init__(self, num_classes=1000): # <======== modificaition to comply fast.ai

        super().__init__()

        

        self.conv = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=64),

            ConvBlock(in_channels=64, out_channels=128),

            ConvBlock(in_channels=128, out_channels=256),

            ConvBlock(in_channels=256, out_channels=512),

        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # <======== modificaition to comply fast.ai

        self.fc = nn.Sequential(

            nn.Dropout(0.2),

            nn.Linear(512, 128),

            nn.PReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.1),

            nn.Linear(128, num_classes),

        )



    def forward(self, x):

        x = self.conv(x)

        #x = torch.mean(x, dim=3)   # <======== modificaition to comply fast.ai

        #x, _ = torch.max(x, dim=2) # <======== modificaition to comply fast.ai

        x = self.avgpool(x)         # <======== modificaition to comply fast.ai

        x = self.fc(x)

        return x
DATA = Path('../input/freesound-audio-tagging-2019')

PREPROCESSED = Path('../input/fat2019_prep_mels1')

WORK = Path('work')

Path(WORK).mkdir(exist_ok=True, parents=True)



CSV_TRN_CURATED = DATA/'train_curated.csv'

CSV_TRN_NOISY = DATA/'train_noisy.csv'

CSV_TRN_NOISY_BEST50S = PREPROCESSED/'trn_noisy_best50s.csv'

CSV_SUBMISSION = DATA/'sample_submission.csv'



MELS_TRN_CURATED = PREPROCESSED/'mels_train_curated.pkl'

MELS_TRN_NOISY = PREPROCESSED/'mels_train_noisy.pkl'

MELS_TRN_NOISY_BEST50S = PREPROCESSED/'mels_trn_noisy_best50s.pkl'

MELS_TEST = PREPROCESSED/'mels_test.pkl'



trn_curated_df = pd.read_csv(CSV_TRN_CURATED)

trn_noisy_df = pd.read_csv(CSV_TRN_NOISY)

trn_noisy50s_df = pd.read_csv(CSV_TRN_NOISY_BEST50S)

test_df = pd.read_csv(CSV_SUBMISSION)



#df = pd.concat([trn_curated_df, trn_noisy_df], ignore_index=True) # not enough memory

df = pd.concat([trn_curated_df, trn_noisy50s_df], ignore_index=True, sort=True)

test_df = pd.read_csv(CSV_SUBMISSION)



X_train = pickle.load(open(MELS_TRN_CURATED, 'rb')) + pickle.load(open(MELS_TRN_NOISY_BEST50S, 'rb'))
from fastai import *

from fastai.vision import *

from fastai.vision.data import *

import random



CUR_X_FILES, CUR_X = list(df.fname.values), X_train



def open_fat2019_image(fn, convert_mode, after_open)->Image:

    # open

    idx = CUR_X_FILES.index(fn.split('/')[-1])

    x = PIL.Image.fromarray(CUR_X[idx])

    # crop

    time_dim, base_dim = x.size

    crop_x = random.randint(0, time_dim - base_dim)

    x = x.crop([crop_x, 0, crop_x+base_dim, base_dim])    

    # standardize

    return Image(pil2tensor(x, np.float32).div_(255))



vision.data.open_image = open_fat2019_image
tfms = get_transforms(do_flip=True, max_rotate=0, max_lighting=0.1, max_zoom=0, max_warp=0.)

src = (ImageList.from_csv(WORK, Path('..')/CSV_TRN_CURATED, folder='trn_curated')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim=',')

)

data = (src.transform(tfms, size=128)

        .databunch(bs=64).normalize(imagenet_stats)

)
data.show_batch(3)
def borrowed_model(pretrained=False, **kwargs):

    return Classifier(**kwargs)



f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(data, borrowed_model, pretrained=False, metrics=[f_score])

learn.unfreeze()



learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-6, 1e-1))
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(100, 3e-3)
learn.save('fat2019_fastai_cnn2d_stage-2')

learn.export()
del X_train

X_test = pickle.load(open(MELS_TEST, 'rb'))

CUR_X_FILES, CUR_X = list(test_df.fname.values), X_test



test = ImageList.from_csv(WORK, Path('..')/CSV_SUBMISSION, folder='test')

learn = load_learner(WORK, test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
test_df[learn.data.classes] = preds

test_df.to_csv('submission.csv', index=False)

test_df.head()
del X_test

X_train = pickle.load(open(MELS_TRN_CURATED, 'rb'))



CUR_X_FILES, CUR_X = list(df.fname.values), X_train

learn = cnn_learner(data, borrowed_model, pretrained=False, metrics=[f_score])

learn.load('fat2019_fastai_cnn2d_stage-2');
# Thanks to https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb

from fastai.callbacks.hooks import *



def visualize_cnn_by_cam(learn, data_index):

    x, _y = learn.data.valid_ds[data_index]

    y = _y.data

    if not isinstance(y, (list, np.ndarray)): # single label -> one hot encoding

        y = np.eye(learn.data.valid_ds.c)[y]



    m = learn.model.eval()

    xb,_ = learn.data.one_item(x)

    xb_im = Image(learn.data.denorm(xb)[0])

    xb = xb.cuda()



    def hooked_backward(cat):

        with hook_output(m[0]) as hook_a: 

            with hook_output(m[0], grad=True) as hook_g:

                preds = m(xb)

                preds[0,int(cat)].backward()

        return hook_a,hook_g

    def show_heatmap(img, hm, label):

        _,axs = plt.subplots(1, 2)

        axs[0].set_title(label)

        img.show(axs[0])

        axs[1].set_title(f'CAM of {label}')

        img.show(axs[1])

        axs[1].imshow(hm, alpha=0.6, extent=(0,img.shape[0],img.shape[0],0),

                      interpolation='bilinear', cmap='magma');

        plt.show()



    for y_i in np.where(y > 0)[0]:

        hook_a,hook_g = hooked_backward(cat=y_i)

        acts = hook_a.stored[0].cpu()

        grad = hook_g.stored[0][0].cpu()

        grad_chan = grad.mean(1).mean(1)

        mult = (acts*grad_chan[...,None,None]).mean(0)

        show_heatmap(img=xb_im, hm=mult, label=str(learn.data.valid_ds.y[data_index]))



for idx in range(10):

    visualize_cnn_by_cam(learn, idx)