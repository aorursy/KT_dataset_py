# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from fastai.conv_learner import *
from fastai.dataset import *
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark=True
im_dir = Path('../input/train/')
error_mask=[]
df_pred = pd.read_csv('../input/train.csv', index_col=[0])
df_pred.fillna('', inplace=True)
df_pred['suspicious'] = False

for index, row in df_pred.iterrows():
    encoded_mask = row['rle_mask'].split(' ')
    if len(encoded_mask) > 1 and len(encoded_mask) < 5 and int(encoded_mask[1]) % 101 == 0:
        df_pred.loc[index,'suspicious'] = True
        error_mask.append(index+".png")

print(len(error_mask))
!nvidia-smi
PATH = './'
TRAIN = '../input/train/images'
TEST = '../input/test/images'
SEGMENTATION = '../input/train.csv'

train_names = [f for f in os.listdir(TRAIN)]
test_names = [f for f in os.listdir(TEST)]
tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)

df = pd.read_csv(SEGMENTATION).set_index('id')
df.head()
class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('id')
        self.segmentation_df.head()
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        if(self.path == TEST): return 0
        masks = self.segmentation_df.loc[self.fnames[i][:-4]]['rle_mask']
        if(type(masks) == float): return 0 #NAN - no ship 
        else: return 1
    
    def get_c(self): return 2 #number of classes
def get_data(sz,bs):
    #data augmentation
    aug_tfms = [RandomRotate(10, tfm_y=TfmType.NO),
                RandomFlip(tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    #md.is_multi = False
    return md
sz = 128 #image size
bs = 32  #batch size
nw = 4   #number of workers for data loader
arch = resnet34 #specify target architecture

md = get_data(sz,bs)
learn = ConvLearner.pretrained(arch, md, ps=0.5) #dropout 50%
learn.opt_fn = optim.Adam
learn.lr_find()
learn.sched.plot()
lr=0.001
learn.fit(lr, 1, cycle_len=5, use_clr=(5,5))
learn.unfreeze()
lr=np.array([1e-4,5e-4,2e-2])
learn.fit(lr, 1, cycle_len=200, use_clr=(20,5))
learn.sched.plot_lr()
learn.sched.plot_loss()
learn.save('Resnet34_tgs_salt_128')