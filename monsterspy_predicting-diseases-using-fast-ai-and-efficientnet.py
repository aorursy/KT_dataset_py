import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # additional plotting functionality



import os

print(os.listdir("../input/data/"))



from cv2 import imread, createCLAHE

import os

import cv2

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import random

from sklearn.utils import shuffle

import scipy as sp

from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

import skimage



# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')





import fastai

from fastai import *

from fastai.vision import *

from fastai.callbacks import *



print(fastai.__version__)
def set_seed(seed=42):

    

    import random

    random.seed(seed)



    # pytorch RNGs

    import torch

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



    # numpy RNG

    import numpy as np

    np.random.seed(seed)

    

set_seed()
df = pd.read_csv('../input/data/Data_Entry_2017.csv')

num_obs = len(df)

print('Number of observations:',num_obs)

df.head(5)
from glob import glob

my_glob = glob('../input/data/images*/images/*.png')

print('Number of Observations: ', len(my_glob))
full_img_paths = {os.path.basename(x): x for x in my_glob}

df['full_path'] = df['Image Index'].map(full_img_paths.get)
df.shape
df['Finding Labels'].nunique()
train_val_list = pd.read_csv('../input/data/train_val_list.txt', header=None, names = ['image_list'])

test_list = pd.read_csv('../input/data/test_list.txt', header=None, names = ['image_list'])
cols = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 

'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

mapp = {}

for idx in range(len(cols)):

    mapp[cols[idx]] = idx
def multi_label(d_multi):

    label = ""

    arr = d_multi.split('|')

    for d in arr:

        if d == 'No Finding':

            label = label + "|"

            continue

        label = label + '|' + str(mapp[d])

    return label[1:]

label_enc = df['Finding Labels'].apply(multi_label)

df['label'] = label_enc
df['name'] = df.full_path.apply(lambda x : "/".join(x.split('/')[2:]))
train = df[df['Image Index'].isin(train_val_list['image_list'].values)].reset_index(drop=True)

test = df[df['Image Index'].isin(test_list['image_list'].values)].reset_index(drop=True)

train.shape, test.shape
train = train[['name', 'label']]

test = test[['name', 'label']]
def clahe(path, augmentations = True):

    img = cv2.imread(path, 1) 

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(4,4))

    planes = cv2.split(img)

    for i in range(0,3):

        planes[i] =clahe.apply(planes[i])

    img = cv2.merge(planes)

    img = cv2.bilateralFilter(img,8,80,80)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
class clahe_bilateral(ImageList):

    def open(self, fn:PathOrStr)->Image:

        img = clahe(fn.replace('/./','').replace('//','/'))

        return vision.Image(px=pil2tensor(img, np.float32).div_(255))
PATH = '../input/'
tfms = get_transforms(do_flip=True, flip_vert=False, xtra_tfms=rand_resize_crop(224))
src = (ImageList.from_df(path=PATH, df=train)

        .split_by_rand_pct()

        .label_from_df(cols='label', label_delim="|")

        .add_test(ImageList.from_df(path=PATH, df=test))

       )
data = (src.transform(tfms, size=254, resize_method=ResizeMethod.SQUISH)

        .databunch(bs=16).normalize(imagenet_stats))

data.path = pathlib.Path('.')
!pip install efficientnet-pytorch
from efficientnet_pytorch import EfficientNet

model =  EfficientNet.from_pretrained('efficientnet-b3', num_classes=14)
opt_func = partial(optim.Adam, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

save_model = partial(SaveModelCallback, monitor='valid_loss', name='bestmodel')

early_stop = partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.005, patience=8)

reduceLR = partial(ReduceLROnPlateauCallback, monitor = 'valid_loss', mode = 'auto', patience = 5, factor = 0.2, min_delta = 0.005)

learn = Learner(

    data,

    model,

    loss_func=nn.BCEWithLogitsLoss(size_average=True),

    opt_func=opt_func,

    callback_fns=[BnFreeze, save_model, early_stop, reduceLR], path = '/kaggle/working', model_dir = '/kaggle/working',

)

learn = learn.split([learn.model._conv_stem,learn.model._blocks,learn.model._conv_head])
learn.freeze()
#learn.lr_find()
#learn.recorder.plot()
#learn.fit_one_cycle(2, 2e-2)
#learn.unfreeze()
#learn.fit_one_cycle(15, max_lr=slice(1e-5, 1e-3), div_factor=10, wd=1e-3)
#preds, _ = learn.TTA(ds_type=DatasetType.Test)

#preds.shape
#pd.DataFrame(preds.numpy()).to_csv('test_preds.csv', index=False)