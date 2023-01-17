%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision import *

from fastai.metrics import error_rate

from fastai import *

from fastai.vision import *

import torch

import torch.nn as nn

import torch.nn.functional as F

from pathlib import Path

import random

import os
bs = 64
path = Path("../input/utkface-new/utkface_aligned_cropped")

path_test=('../input/utkface-new/UTKface_Aligned_cropped/crop_part1')

path
path.ls()
im =PIL.Image.open(path/'UTKFace/26_1_1_20170117201805678.jpg.chip.jpg')
im
path_labels = path

path_img = path/'UTKFace'
fnames = get_image_files(path_img)

fnames[:5]
def extract_age(filename):

    return float(filename.stem.split('_')[0])
extract_age(path_img/'28_0_0_20170117202521375.jpg.chip.jpg')
np.random.seed(2)

ds=get_transforms(do_flip=False,max_rotate=0,max_zoom=1, max_lighting=0, max_warp=0, p_affine=0, p_lighting=0 )
fn_paths = path_img.ls(); fn_paths[:2]
def extract_age(filename):

    return float(filename.stem.split('_')[0])
def load_face_data(img_size, batch_size,path):

    tfms = get_transforms(max_warp=0.)

    return (ImageList.from_folder(path)

            .random_split_by_pct(0.2, seed=666)

            .label_from_func(extract_age)

            .transform(tfms, size=img_size)

            .databunch(bs=batch_size))
data = load_face_data(224, 256,path)
data.show_batch(rows=3, figsize=(7,7))
age=[extract_age(i) for i in path_img.ls()]

plt.figure(figsize=(10, 5))

plt.plot(*zip(*sorted(Counter(age).items())), '.:')

plt.title('Number of Images by Age')

plt.ylabel('count')

plt.xlabel('age')

plt.grid()
class AgeModel(nn.Module):

    def __init__(self):

        super().__init__()

        layers = list(models.resnet34(pretrained=True).children())[:-2]

        layers += [AdaptiveConcatPool2d(),Flatten()]

        layers += [nn.Linear(1024, 16), nn.ReLU(), nn.Linear(16,1)]

        self.agemodel = nn.Sequential(*layers)

    def forward(self, x):

        return self.agemodel(x).squeeze() # could add 116*torch.sigmoid
model = AgeModel()
learn = Learner(data, model, loss_func = F.l1_loss, model_dir="/tmp/model/")

learn=learn.load("/kaggle/input/training/ageTraining")

learn.split([model.agemodel[2],model.agemodel[-3]])
learn.layer_groups[-1]

learn.freeze_to(-1)
learn.fit_one_cycle(1)

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6,max_lr =1e-3)
learn.unfreeze()



learn.lr_find(); learn.recorder.plot()
learn.fit_one_cycle(12, max_lr = slice(1e-6,1e-5))
learn.save('/kaggle/working/ageTraining')
Path('/kaggle/working').ls()
os.chdir(r'/kaggle/working')

from IPython.display import FileLink

FileLink(r'ageTraining.pth')
learn.show_results(rows=4)
img = data.train_ds[0][0]

img


learn.model.eval()
data1 = learn.data.train_ds[0][0]

data1
pred = learn.predict(data)
